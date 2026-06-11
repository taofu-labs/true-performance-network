import asyncio
import time

import bittensor as bt
import numpy as np
from loguru import logger

from common import settings as common_settings
from common.chain import get_subtensor, get_wallet
from validator.validator_health import HealthServerMixin
from validator import settings as validator_settings


class Validator(HealthServerMixin):
    def __init__(self, coldkey: str | None = None, wallet_hotkey: str | None = None, wallet=None):
        super().__init__()
        self.wallet = wallet or get_wallet(
            coldkey=coldkey,
            hotkey=wallet_hotkey,
            wallet_path=validator_settings.WALLET_PATH,
        )
        self.hotkey = self.wallet.hotkey.ss58_address
        self.available: bool = True

        # Task crash recovery state
        self._tasks_failed: int = 0
        self._last_heartbeat: float = time.time()

        from validator.storage import PersistentSet, validator_storage_dir
        self._scored: PersistentSet = PersistentSet(validator_storage_dir() / "scored.json")

        logger.info(
            f"Running Validator. Bittensor: {common_settings.BITTENSOR}. "
            f"Network: {common_settings.NETWORK}. Netuid: {common_settings.NETUID}"
        )
        self.subtensor = get_subtensor()

        self.metagraph = bt.Metagraph(netuid=common_settings.NETUID, lite=False, network=common_settings.NETWORK)

        if common_settings.BITTENSOR:
            try:
                self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            except ValueError:
                logger.warning(
                    f"Hotkey {self.wallet.hotkey.ss58_address} not registered on subnet {common_settings.NETUID} "
                    f"(network: {common_settings.NETWORK})"
                )

    async def _validator_loop(self):
        """TPN validator loop — scans chain, scores models, sets weights."""
        logger.info(f"🔄 Starting TPN validator loop for {self.hotkey[:8]}")

        while True:
            try:
                current_block = self.subtensor.get_current_block()
                self.metagraph.sync(subtensor=self.subtensor, lite=False)

                from competition.github_config import get_active_competitions
                from common.models.competition import CompetitionPhase

                specs = get_active_competitions(index_url=common_settings.COMPETITION_INDEX_URL, current_block=current_block)

                if not specs:
                    logger.info("No active competitions. Sleeping.")
                    continue

                for spec in specs:
                    phase = spec.phase(current_block)
                    remaining = spec.blocks_until_next_phase(current_block)
                    logger.info(f"Block {current_block} | {spec.id} | {phase.value} | {remaining} blocks left")

                    if phase == CompetitionPhase.OPEN:
                        logger.debug(f"OPEN phase — waiting for commit_end_block {spec.commit_end_block}")

                    elif phase == CompetitionPhase.SCORING:
                        if current_block >= spec.scoring_starts_at() and spec.id not in self._scored:
                            await self._run_scoring(spec)
                            self._scored.add(spec.id)

                    elif phase == CompetitionPhase.COMPLETE:
                        if spec.id not in self._scored:
                            await self._run_scoring(spec)
                            self._scored.add(spec.id)

            except Exception as e:
                logger.exception(f"Validator loop error: {e}")
            finally:
                await asyncio.sleep(validator_settings.VALIDATOR_LOOP_INTERVAL)

    async def _run_scoring(self, spec):
        from validator.chain_scanner import scan_reveals
        from validator.scorer import score_model
        from competition.scoring import sort_by_self_reported, compute_emission_weights

        logger.info(f"🏆 Scoring competition {spec.id}")

        reveals = scan_reveals(self.subtensor, spec)
        if not reveals:
            logger.warning("No valid reveals. Skipping weight update.")
            return

        # Filter: only hotkeys registered on subnet
        registered = set(self.metagraph.hotkeys)
        reveals = {hk: sub for hk, sub in reveals.items() if hk in registered}
        if not reveals:
            logger.warning("No reveals from registered hotkeys. Skipping weight update.")
            return

        top_n = sort_by_self_reported(reveals, spec)[: spec.top_n]
        results = [score_model(hotkey, submission, spec) for hotkey, submission in top_n]
        ranked = sorted(results, key=lambda r: r.final_score, reverse=True)
        hotkey_weights = compute_emission_weights(ranked, spec.emission_distribution)

        # Convert hotkey → UID for set_weights (process_weights_for_netuid expects int UIDs)
        hotkey_to_uid = {hk: int(uid) for uid, hk in zip(self.metagraph.uids.tolist(), self.metagraph.hotkeys)}
        uid_weights = {
            hotkey_to_uid[hk]: w
            for hk, w in hotkey_weights.items()
            if hk in hotkey_to_uid
        }

        logger.info(f"Weights: {[(k[:8], v) for k, v in hotkey_weights.items() if v > 0]}")
        await self.set_weights(weights=uid_weights)

        # Registry publication — only runs if HF_TOKEN and HF_ORG are configured
        from common import settings as common_settings
        if common_settings.HF_TOKEN and common_settings.HF_ORG:
            logger.info(f"Publishing registry for {spec.id} to {common_settings.HF_ORG}")
            # TODO: implement registry publication
        else:
            logger.debug("HF_TOKEN/HF_ORG not set — skipping registry publication")

    async def weight_loop(self):
        """Weight loop — periodically copies weights from chain as fallback."""
        loop_count = 0
        logger.info(f"🔄 Starting weight loop for validator {self.hotkey[:8]}")

        while True:
            loop_count += 1
            try:
                logger.debug(f"Weight loop iteration {loop_count} starting")
                self.metagraph.sync(subtensor=self.subtensor, lite=False)
                logger.debug("VALIDATOR: WEIGHT LOOP RUNNING — copying weights from chain")
                await self.set_weights(weights=self.copy_weights_from_chain())
                logger.debug(f"Weight loop iteration {loop_count} completed successfully")

            except Exception as e:
                logger.exception(f"Error in weight loop iteration {loop_count}: {e}")
                try:
                    await self.set_weights(weights=self.copy_weights_from_chain())
                except Exception:
                    pass

            finally:
                logger.info(
                    f"💤 Weight submission loop sleeping for {validator_settings.WEIGHT_SUBMIT_INTERVAL} seconds 💤"
                )
                await asyncio.sleep(validator_settings.WEIGHT_SUBMIT_INTERVAL)

    async def run_validator(self):
        logger.info("🚀 Starting validator")
        # Start the healthcheck server
        if validator_settings.LAUNCH_HEALTH:
            await self._start_health_server()
            logger.info(f"🏥 Health server started for validator {self.hotkey[:8]}")
        else:
            logger.warning(
                "⚠️ Validator healthcheck API not configured in settings (VALIDATOR_HEALTH_PORT missing). Skipping."
            )

        # Task management state
        self._weight_task = None
        self._validator_task = None
        task_restart_count = {"weight_loop": 0, "validator_loop": 0}
        max_restarts = 10
        restart_delay = 5
        status_log_interval = 300
        last_status_log = 0

        # Main task monitoring loop
        while True:
            try:
                current_time = time.time()

                # Log task status periodically
                if current_time - last_status_log > status_log_interval:
                    self._log_task_status(
                        weight_task=self._weight_task,
                        validator_task=self._validator_task,
                        task_restart_count=task_restart_count,
                    )
                    last_status_log = current_time

                # Create tasks if they don't exist or have completed/failed
                if self._weight_task is None or self._weight_task.done():
                    if self._weight_task is not None and self._weight_task.done():
                        try:
                            self._weight_task.result()
                            logger.info("Weight loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Weight loop task failed: {e}")
                            task_restart_count["weight_loop"] += 1

                            if task_restart_count["weight_loop"] >= max_restarts:
                                logger.critical(f"Weight loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Weight loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting weight loop task (attempt {task_restart_count['weight_loop'] + 1})"
                    )
                    self._weight_task = asyncio.create_task(self.weight_loop())

                if self._validator_task is None or self._validator_task.done():
                    if self._validator_task is not None and self._validator_task.done():
                        try:
                            self._validator_task.result()
                            logger.info("Validator loop task completed normally")
                        except Exception as e:
                            logger.exception(f"❌ Validator loop task failed: {e}")
                            task_restart_count["validator_loop"] += 1

                            if task_restart_count["validator_loop"] >= max_restarts:
                                logger.critical(f"Validator loop has failed {max_restarts} times, giving up")
                                raise Exception(f"Validator loop exceeded maximum restart attempts ({max_restarts})")

                    logger.info(
                        f"🔄 Starting/restarting validator loop task (attempt {task_restart_count['validator_loop'] + 1})"
                    )
                    self._validator_task = asyncio.create_task(self._validator_loop())

                # Wait for either task to complete (indicating failure since they run forever)
                logger.debug("🔍 Monitoring tasks for failures...")
                done, pending = await asyncio.wait(
                    [self._weight_task, self._validator_task], return_when=asyncio.FIRST_COMPLETED
                )

                for task in done:
                    if task == self._weight_task:
                        logger.warning("⚠️ Weight loop task completed unexpectedly")
                    elif task == self._validator_task:
                        logger.warning("⚠️ Validator loop task completed unexpectedly")

                if restart_delay > 0:
                    logger.info(f"⏳ Waiting {restart_delay} seconds before restarting failed tasks...")
                    await asyncio.sleep(restart_delay)

            except Exception as e:
                logger.exception(f"Critical error in validator task manager: {e}")

                if self._weight_task and not self._weight_task.done():
                    self._weight_task.cancel()
                    try:
                        await self._weight_task
                    except asyncio.CancelledError:
                        pass

                if self._validator_task and not self._validator_task.done():
                    self._validator_task.cancel()
                    try:
                        await self._validator_task
                    except asyncio.CancelledError:
                        pass

                self._weight_task = None
                self._validator_task = None

                await asyncio.sleep(10)

    async def set_weights(self, weights: dict[int, float]):
        """Sets the validator weights to the metagraph hotkeys."""
        logger.info("Attempting to set weights to Bittensor.")
        if not common_settings.BITTENSOR:
            logger.warning("Bittensor is not enabled, skipping weight submission")
            return

        if not hasattr(self, "wallet") or not self.wallet:
            logger.warning("Wallet not initialized, skipping weight submission")
            return

        if not hasattr(self, "subtensor") or not self.subtensor:
            logger.warning("Subtensor not initialized, skipping weight submission")
            return

        if not hasattr(self, "metagraph") or not self.metagraph:
            logger.warning("Metagraph not initialized, skipping weight submission")
            return

        try:
            uids, scores = zip(*weights.items())
            uids = np.array(uids)
            scores = np.array(scores)

            if np.isnan(scores).any():
                logger.warning("Scores contain NaN values. Replacing with 0.")
                scores = np.nan_to_num(scores, 0)

            if np.sum(scores) == 0:
                logger.warning("All scores are zero, skipping weight submission")
                return

            (
                processed_weight_uids,
                processed_weights,
            ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids=uids,
                weights=scores,
                netuid=int(common_settings.NETUID),
                subtensor=self.subtensor,
                metagraph=self.metagraph,
            )

            weight_dict = dict(zip(processed_weight_uids.tolist(), processed_weights.tolist()))
            logger.info(f"Setting weights for {len(weight_dict)} miners")
            logger.debug(f"Weight details: {weight_dict}")

            success, response = self.subtensor.set_weights(
                wallet=self.wallet,
                netuid=int(common_settings.NETUID),
                uids=processed_weight_uids,
                weights=processed_weights,
                wait_for_finalization=False,
                version_key=common_settings.__SPEC_VERSION__,
            )

            if success:
                logger.success("Successfully submitted weights to Bittensor.")
                logger.debug(f"Response: {response}")
            else:
                logger.error("Failed to submit weights to Bittensor")
                logger.error(f"Response: {response}")

        except Exception as e:
            logger.exception(f"Error submitting weights to Bittensor: {e}")

    def copy_weights_from_chain(self) -> dict[int, float]:
        """Copy weights from the chain to the validator."""
        self.metagraph.sync(subtensor=self.subtensor, lite=False)

        valid_indices = np.where(self.metagraph.validator_permit)[0]
        valid_weights = self.metagraph.weights[valid_indices]
        valid_stakes = self.metagraph.stake[valid_indices]
        normalized_stakes = valid_stakes / np.sum(valid_stakes)
        stake_weighted_average = np.dot(normalized_stakes, valid_weights).astype(float).tolist()

        if len(self.metagraph.uids) == 0:
            logger.warning("No valid indices found in metagraph, returning empty weights")
            return {}

        return dict(zip(self.metagraph.uids, list(stake_weighted_average)))

    async def get_validator_status(self) -> dict:
        """Get current validator status for monitoring, including task states."""
        status = {
            "hotkey": self.hotkey[:8] if hasattr(self, "hotkey") else "N/A",
            "available": self.available,
            "last_heartbeat": self._last_heartbeat,
            "uptime": time.time() - self._last_heartbeat if self._last_heartbeat > 0 else 0,
        }

        if hasattr(self, "_weight_task") and self._weight_task:
            status["weight_task_running"] = not self._weight_task.done()
            status["weight_task_cancelled"] = self._weight_task.cancelled()
        else:
            status["weight_task_running"] = False

        if hasattr(self, "_validator_task") and self._validator_task:
            status["validator_task_running"] = not self._validator_task.done()
            status["validator_task_cancelled"] = self._validator_task.cancelled()
        else:
            status["validator_task_running"] = False

        return status

    def _log_task_status(self, weight_task: asyncio.Task, validator_task: asyncio.Task, task_restart_count: dict):
        """Log the current status of both tasks for debugging."""
        weight_status = "None"
        if weight_task:
            if weight_task.done():
                weight_status = "Done/Failed"
            elif weight_task.cancelled():
                weight_status = "Cancelled"
            else:
                weight_status = "Running"

        validator_status = "None"
        if validator_task:
            if validator_task.done():
                validator_status = "Done/Failed"
            elif validator_task.cancelled():
                validator_status = "Cancelled"
            else:
                validator_status = "Running"

        logger.info(
            f"📊 Task Status - Weight: {weight_status} (restarts: {task_restart_count['weight_loop']}), "
            f"Validator: {validator_status} (restarts: {task_restart_count['validator_loop']})"
        )


if __name__ == "__main__":
    gradient_validator = Validator(
        coldkey=validator_settings.WALLET_COLDKEY, wallet_hotkey=validator_settings.WALLET_HOTKEY
    )
    asyncio.run(gradient_validator.run_validator())
