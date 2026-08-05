import asyncio
import time

import bittensor as bt
import numpy as np
from loguru import logger

from common import settings as common_settings
from common.chain import get_subtensor, get_wallet
from validator.validator_health import HealthServerMixin
from validator.api import LeaderApiMixin
from validator import settings as validator_settings
from validator import store


class Validator(HealthServerMixin, LeaderApiMixin):
    def __init__(
        self,
        coldkey: str | None = None,
        wallet_hotkey: str | None = None,
        wallet=None,
        subtensor=None,
        metagraph=None,
    ):
        super().__init__()

        self.mode = common_settings.VALIDATOR_MODE
        if self.mode not in ("leader", "follower"):
            raise ValueError(f"VALIDATOR_MODE must be 'leader' or 'follower', got {self.mode!r}")

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

        self._db = store.init_db()

        if self.mode == "follower":
            from validator.leader_client import LeaderClient
            self._leader_client = LeaderClient(validator_settings.LEADER_VALIDATOR_URL)

        logger.info(
            f"Running Validator (mode={self.mode}). Bittensor: {common_settings.BITTENSOR}. "
            f"Network: {common_settings.NETWORK}. Netuid: {common_settings.NETUID}"
        )
        self.subtensor = subtensor
        self.metagraph = metagraph
        if common_settings.BITTENSOR:
            self.subtensor = subtensor or get_subtensor()
            self.metagraph = metagraph or self.subtensor.subnets.metagraph(common_settings.NETUID)

        if common_settings.BITTENSOR and self.metagraph is not None:
            try:
                self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
            except ValueError:
                logger.warning(
                    f"Hotkey {self.wallet.hotkey.ss58_address} not registered on subnet {common_settings.NETUID} "
                    f"(network: {common_settings.NETWORK})"
                )

    def _get_active_competitions(self, current_block: int):
        """Active competitions from this validator's own source of truth.

        Leader reads its local SQLite store directly (it owns the data).
        Follower fetches from the leader's public /v1/competitions API.
        """
        from common.models.competition import CompetitionSpec

        if self.mode == "leader":
            specs = [CompetitionSpec.model_validate(s) for s in store.list_competitions(self._db)]
            return [s for s in specs if s.is_active(current_block)]

        from competition.leader_config_client import get_active_competitions
        return get_active_competitions(
            base_url=validator_settings.LEADER_VALIDATOR_URL, current_block=current_block
        )

    async def _leader_loop(self):
        """Leader loop — scans chain, scores models, persists results, sets weights."""
        logger.info(f"🔄 Starting leader loop for {self.hotkey[:8]}")

        while True:
            try:
                current_block = self.subtensor.block()
                self.metagraph = self.subtensor.subnets.metagraph(common_settings.NETUID)

                from common.models.competition import CompetitionPhase

                specs = self._get_active_competitions(current_block)

                if not specs:
                    logger.info("No active competitions. Sleeping.")
                    continue

                for spec in specs:
                    phase = spec.phase(current_block)

                    if phase == CompetitionPhase.COMPLETE:
                        continue

                    remaining = spec.blocks_until_next_phase(current_block)
                    logger.info(f"Block {current_block} | {spec.id} | {phase.value} | {remaining} blocks left")

                    if phase == CompetitionPhase.OPEN:
                        logger.debug(f"OPEN phase — waiting for commit_end_block {spec.commit_end_block}")

                    elif phase == CompetitionPhase.REVEALING:
                        logger.debug(f"REVEALING phase — waiting for reveal grace to end at {spec.scoring_starts_at()}")

                    elif phase == CompetitionPhase.SCORING:
                        if not store.is_scored(self._db, spec.id):
                            done = await self._run_scoring(spec, current_block)
                            if done:
                                store.mark_scored(self._db, spec.id)
                            else:
                                attempts = store.bump_reveal_attempts(self._db, spec.id)
                                if attempts >= store.MAX_REVEAL_ATTEMPTS:
                                    logger.warning(
                                        f"{spec.id}: giving up after {attempts} attempts — marking scored with no result"
                                    )
                                    store.mark_scored(self._db, spec.id, status="failed_no_reveals")
                                else:
                                    logger.info(
                                        f"{spec.id}: retryable scoring failure, attempt {attempts}/{store.MAX_REVEAL_ATTEMPTS}"
                                    )

            except Exception as e:
                logger.exception(f"Leader loop error: {e}")
            finally:
                await asyncio.sleep(validator_settings.VALIDATOR_LOOP_INTERVAL)

    async def _follower_loop(self):
        """Follower loop — parses competitions, reads scoring results from the leader, sets weights."""
        logger.info(f"🔄 Starting follower loop for {self.hotkey[:8]} (leader={validator_settings.LEADER_VALIDATOR_URL})")

        while True:
            try:
                self.metagraph = self.subtensor.subnets.metagraph(common_settings.NETUID)

                from common.models.competition import CompetitionPhase
                from competition.scoring import compute_emission_weights
                from common.models.submission import ScoringResult

                current_block = self.subtensor.block()
                specs = self._get_active_competitions(current_block)

                for spec in specs:
                    if store.is_scored(self._db, spec.id):
                        continue
                    phase = spec.phase(current_block)
                    if phase not in (CompetitionPhase.SCORING, CompetitionPhase.DISTRIBUTING, CompetitionPhase.COMPLETE):
                        continue

                    runs, scored_status = self._leader_client.get_scoring_results(spec.id)
                    if not runs:
                        if scored_status == "failed_no_reveals":
                            logger.warning(f"{spec.id}: leader gave up with no reveals — marking scored, no weights")
                            store.mark_scored(self._db, spec.id, status="failed_no_reveals")
                        else:
                            logger.debug(f"{spec.id}: no scoring results from leader yet")
                        continue

                    latest = runs[-1]
                    ranked = sorted(
                        [ScoringResult(**r) for r in latest["results"]],
                        key=lambda r: r.final_score,
                        reverse=True,
                    )
                    hotkey_weights = compute_emission_weights(ranked, spec.emission_distribution)

                    logger.info(f"{spec.id}: following leader run {latest['run_id']} — weights: "
                                f"{[(k[:8], v) for k, v in hotkey_weights.items() if v > 0]}")
                    store.record_weights(self._db, spec.id, hotkey_weights)
                    store.mark_scored(self._db, spec.id)

            except Exception as e:
                logger.exception(f"Follower loop error: {e}")
            finally:
                await asyncio.sleep(validator_settings.FOLLOWER_POLL_INTERVAL)

    async def _run_scoring(self, spec, current_block: int) -> bool:
        """Returns True if scoring reached a terminal state (success, or a
        non-retryable abort) and the competition should be marked scored.
        Returns False for a retryable failure (no reveals yet, coordinator
        unreachable) so the caller can retry on a later tick."""
        from validator.chain_scanner import scan_reveals
        from validator.scorer import precheck_one, benchmark_one, dedup_winner, _skip, PrecheckPass, ScoringOutcome
        from competition.benchmark_client import make_coordinator
        from competition.precheck_client import PrecheckContainer
        from competition.scoring import sort_by_self_reported, compute_emission_weights
        from validator import settings as validator_settings

        logger.info(f"🏆 Scoring competition {spec.id}")

        all_outcomes: list[ScoringOutcome] = []

        reveals = scan_reveals(self.subtensor, spec, self._db)
        logger.debug(f"scan_reveals returned {len(reveals)} reveal(s): {list(reveals.keys())}")
        if not reveals:
            logger.warning("No valid reveals. Skipping weight update.")
            return False

        registered = set(self.metagraph.hotkeys)
        reveals = {hk: sub for hk, sub in reveals.items() if hk in registered}
        logger.debug(f"{len(reveals)} reveal(s) from registered hotkeys: {list(reveals.keys())}")
        if not reveals:
            logger.warning("No reveals from registered hotkeys. Skipping weight update.")
            return False

        # Collateral skip: a hotkey below COLLATERAL_MIN_THRESHOLD is excluded
        # from this scoring run only — not a ban, no persistent state change.
        # `None` (runtime doesn't report collateral yet) never skips.
        collateral_by_hotkey = {n.hotkey: n.collateral_locked for n in self.metagraph.neurons}
        min_collateral = validator_settings.COLLATERAL_MIN_THRESHOLD
        if min_collateral > 0:
            def _has_collateral(hotkey: str) -> bool:
                collateral = collateral_by_hotkey.get(hotkey)
                if collateral is None:
                    return True
                return collateral.amount >= min_collateral

            skipped = {hk for hk in reveals if not _has_collateral(hk)}
            for hk in skipped:
                logger.info(f"{hk[:12]} skipped — collateral below threshold ({collateral_by_hotkey[hk]} < {min_collateral})")
            reveals = {hk: sub for hk, sub in reveals.items() if hk not in skipped}
            if not reveals:
                logger.warning("No reveals left after collateral skip. Skipping weight update.")
                return False

        # sha256 dedup: when two reveals share the same self-reported file_sha256,
        # only the earlier reveal_block (chain-native, unforgeable) is eligible.
        # Filtered out here, before precheck, so a losing duplicate never costs a
        # Docker container run. Safe to key on self-reported hash: a hotkey that
        # falsely claims another's hash gets banned by precheck's existing
        # sha256-mismatch check if it wins the slot, or is simply skipped if it
        # loses — no incentive either way.
        seen_hashes: dict[str, tuple[str, int]] = {}
        dedup_losers: dict[str, tuple[str, int]] = {}
        for hotkey, (submission, block) in sorted(reveals.items(), key=lambda kv: kv[1][1]):
            loser = dedup_winner(seen_hashes, hotkey, submission.file_sha256, block)
            if loser is not None:
                dedup_losers[hotkey] = loser
            else:
                seen_hashes[submission.file_sha256] = (hotkey, block)

        if dedup_losers:
            for hotkey, (winner_hotkey, winner_block) in dedup_losers.items():
                submission, block = reveals[hotkey]
                reason = (
                    f"duplicate file_sha256={submission.file_sha256[:12]} — earlier reveal by "
                    f"{winner_hotkey[:12]} at block {winner_block} (this reveal at block {block})"
                )
                all_outcomes.append(_skip(hotkey, submission, reason))
                logger.info(f"SKIPPED {hotkey[:12]}: {reason} — dedup loss")
            reveals = {hk: v for hk, v in reveals.items() if hk not in dedup_losers}
            if not reveals:
                logger.warning("No reveals left after dedup filtering. Skipping weight update.")
                return True

        ranked_candidates = sort_by_self_reported(
            {hk: sub for hk, (sub, _block) in reveals.items()}, spec
        )
        logger.debug(f"Ranked candidates by self-reported score: {[hk[:12] for hk, _ in ranked_candidates]}")
        coordinator = make_coordinator()

        try:
            available = await asyncio.to_thread(coordinator.list_benchmarks)
            logger.debug(f"Coordinator available benchmarks: {available}")
            missing = {t.name for t in spec.benchmarks} - available
            if missing:
                logger.error(f"Coordinator missing benchmarks {missing} — aborting scoring")
                return False
        except Exception as e:
            logger.error(f"Cannot reach coordinator: {e} — aborting scoring")
            return False

        precheck_ctr: PrecheckContainer | None = PrecheckContainer(base_repo=spec.model_repo)
        try:
            await asyncio.to_thread(precheck_ctr.start)
        except Exception as e:
            logger.error(f"Precheck container failed to start: {e} — skipping precheck")
            precheck_ctr = None

        # Phase 1: sequential precheck — backfill on skip OR disqualify until top_n pass
        passed: list[PrecheckPass] = []
        try:
            for hotkey, submission in ranked_candidates:
                result = await asyncio.to_thread(precheck_one, hotkey, submission, spec, precheck_ctr, self._db)
                if isinstance(result, PrecheckPass):
                    passed.append(result)
                    if len(passed) == spec.top_n:
                        break
                else:
                    all_outcomes.append(result)
                    logger.info(f"{result.kind.name} {hotkey[:12]}: {result.reason} — backfilling")
        finally:
            if precheck_ctr:
                try:
                    await asyncio.to_thread(precheck_ctr.stop)
                except Exception as e:
                    logger.warning(f"Precheck container stop error: {e}")

        logger.debug(f"Precheck done: {len(passed)} passed, {len(all_outcomes)} skipped/disqualified so far")
        if not passed:
            logger.warning("No submissions passed precheck. Skipping weight update.")
            return True

        # Phase 2: parallel benchmark — all passed miners run concurrently
        poll_interval = getattr(validator_settings, "BENCHMARK_POLL_INTERVAL", 30.0)
        benchmark_outcomes = await asyncio.gather(*[
            benchmark_one(p, spec, coordinator, self._db, poll_interval)
            for p in passed
        ])
        all_outcomes.extend(benchmark_outcomes)
        logger.debug(f"Benchmark done: {len(benchmark_outcomes)} outcome(s)")

        all_results = [o.result for o in all_outcomes]
        ranked = sorted(all_results, key=lambda r: r.final_score, reverse=True)
        hotkey_weights = compute_emission_weights(ranked, spec.emission_distribution)
        logger.debug(f"Final ranking: {[(r.hotkey[:12], r.final_score) for r in ranked]}")
        logger.debug(f"Emission weights: {hotkey_weights}")

        store.record_scoring_run(
            self._db, spec.id, current_block, all_outcomes,
            {hk: sub for hk, (sub, _block) in reveals.items()},
        )
        store.record_weights(self._db, spec.id, hotkey_weights)

        logger.info(f"Weights computed: {[(k[:8], v) for k, v in hotkey_weights.items() if v > 0]}")
        return True

    async def _compute_and_set_aggregate_weights(self, current_block: int) -> bool:
        """
        Find every competition currently in its post-scoring distribution
        window, sum their recorded per-hotkey weights (scaled by each
        competition's emission_weight), and push the result to chain.

        Returns True if weights were set (at least one competition was
        distributing), False if there was nothing to distribute this tick
        (caller should fall back to copy_weights_from_chain in that case).
        """
        from competition.scoring import aggregate_competition_weights

        specs = self._get_active_competitions(current_block)
        distributing = []
        for spec in specs:
            if not spec.is_distributing(current_block):
                continue
            weights = store.latest_weights_for_competition(self._db, spec.id)
            if weights is None:
                logger.warning(f"{spec.id} is distributing but has no recorded weights yet — skipping")
                continue
            distributing.append((spec, weights))

        if not distributing:
            return False

        registered = set(self.metagraph.hotkeys)
        hotkey_weights = aggregate_competition_weights(distributing, registered)

        hotkey_to_uid = {hk: uid for uid, hk in enumerate(self.metagraph.hotkeys)}
        uid_weights = {hotkey_to_uid[hk]: w for hk, w in hotkey_weights.items() if hk in hotkey_to_uid}

        if not uid_weights:
            return False

        logger.info(f"Aggregated weights across {len(distributing)} competitions: "
                    f"{[(k[:8], round(v, 4)) for k, v in hotkey_weights.items() if v > 0]}")

        # bt.set_weights only rescales among submitted uids — it never tops up
        # a shortfall on its own, so any unallocated share must be burned to
        # uid 0 explicitly or the miners present silently inherit it.
        total = sum(uid_weights.values())
        remainder = 1.0 - total
        if remainder > 1e-9:
            uid_weights[0] = uid_weights.get(0, 0.0) + remainder
            logger.info(f"Burning unallocated weight {remainder:.4f} to uid 0")

        await self.set_weights(weights=uid_weights)
        return True

    async def weight_loop(self):
        """Weight loop — pushes aggregated competition weights to chain, falling
        back to copying weights from other validators when nothing is currently
        distributing."""
        loop_count = 0
        logger.info(f"🔄 Starting weight loop for validator {self.hotkey[:8]}")

        while True:
            loop_count += 1
            try:
                logger.debug(f"Weight loop iteration {loop_count} starting")
                self.metagraph = self.subtensor.subnets.metagraph(common_settings.NETUID)
                current_block = self.subtensor.block()

                distributed = await self._compute_and_set_aggregate_weights(current_block)
                if not distributed:
                    logger.debug("No competitions currently distributing — falling back to copy_weights_from_chain")
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
        logger.info(f"🚀 Starting validator (mode={self.mode})")
        # Start the healthcheck server
        if validator_settings.LAUNCH_HEALTH:
            await self._start_health_server()
            logger.info(f"🏥 Health server started for validator {self.hotkey[:8]}")
        else:
            logger.warning(
                "⚠️ Validator healthcheck API not configured in settings (VALIDATOR_HEALTH_PORT missing). Skipping."
            )

        if self.mode == "leader":
            await self._start_leader_api()

        scoring_loop = self._leader_loop if self.mode == "leader" else self._follower_loop

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
                    self._validator_task = asyncio.create_task(scoring_loop())

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

        if not weights:
            logger.warning("No weights to submit, skipping")
            return

        try:
            uids, scores = zip(*weights.items())
            scores = np.array(scores)

            if np.isnan(scores).any():
                logger.warning("Scores contain NaN values. Replacing with 0.")
                scores = np.nan_to_num(scores, 0)

            if np.sum(scores) == 0:
                logger.warning("All scores are zero, skipping weight submission")
                return

            weight_dict = dict(zip(uids, scores.tolist()))
            logger.info(f"Setting weights for {len(weight_dict)} miners")
            logger.debug(f"Weight details: {weight_dict}")

            result = bt.set_weights(
                int(common_settings.NETUID),
                weight_dict,
                wallet=self.wallet,
                network=common_settings.NETWORK,
                version_key=common_settings.__SPEC_VERSION__,
            )

            if result.success:
                logger.success("Successfully submitted weights to Bittensor.")
                logger.debug(f"Response: {result}")
            else:
                logger.error("Failed to submit weights to Bittensor")
                logger.error(f"Response: {result}")

        except Exception as e:
            logger.exception(f"Error submitting weights to Bittensor: {e}")

    def copy_weights_from_chain(self) -> dict[int, float]:
        """Copy weights from the chain to the validator: each validator's submitted
        weight row, stake-weighted-averaged across validators into one consensus
        weight per miner uid."""
        self.metagraph = self.subtensor.subnets.metagraph(common_settings.NETUID)
        validators = self.metagraph.validators

        if not validators:
            logger.warning("No valid indices found in metagraph, returning empty weights")
            return {}

        weight_rows = self.subtensor.weights.weights(int(common_settings.NETUID))
        total_stake = sum(v.total_stake.alpha for v in validators)
        if total_stake == 0:
            logger.warning("Validators have zero total stake, returning empty weights")
            return {}

        consensus: dict[int, float] = {}
        for validator in validators:
            row = weight_rows.get(validator.uid, {})
            stake_fraction = validator.total_stake.alpha / total_stake
            for miner_uid, fraction in row.items():
                consensus[miner_uid] = consensus.get(miner_uid, 0.0) + stake_fraction * fraction

        return consensus

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
