import asyncio
import time
from typing import Optional

import bittensor as bt
import numpy as np
from loguru import logger

from common import settings as common_settings
from common.chain import current_block as get_current_block, get_subtensor, get_wallet
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

        self._last_heartbeat: float = time.time()

        self._db = store.init_db()
        self._coordinator = None

        if self.mode == "leader":
            from competition.precheck_client import cleanup_stale_containers
            in_flight = [
                spec["id"] for spec in store.list_competitions(self._db)
                if store.get_stage(self._db, spec["id"]) == "stage2_scoring"
            ]
            cleanup_stale_containers(keep_competition_ids=in_flight)
            for cid in in_flight:
                reset_count = store.reset_stale_candidate_statuses(self._db, cid)
                if reset_count:
                    logger.info(f"{cid}: reset {reset_count} candidate(s) stuck in 'prechecking' from a previous run")

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

    def _get_coordinator(self):
        if self._coordinator is None:
            from competition.benchmark_client import make_coordinator
            self._coordinator = make_coordinator()
        return self._coordinator

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
                current_block = get_current_block(self.subtensor)
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
                        if store.is_scored(self._db, spec.id):
                            continue
                        stage = store.get_stage(self._db, spec.id)
                        if stage == "stage1_ranking":
                            await self.run_stage_1(spec)
                        elif stage == "stage2_scoring":
                            await self.run_stage_2(spec, current_block)

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

                current_block = get_current_block(self.subtensor)
                specs = self._get_active_competitions(current_block)

                for spec in specs:
                    if store.is_scored(self._db, spec.id):
                        continue
                    phase = spec.phase(current_block)
                    if phase not in (CompetitionPhase.SCORING, CompetitionPhase.DISTRIBUTING, CompetitionPhase.COMPLETE):
                        continue

                    results, scored_status = self._leader_client.get_scoring_results(spec.id)
                    if not results:
                        if scored_status in ("failed_no_reveals", "failed_no_participants", "failed_stage1_infra"):
                            logger.warning(f"{spec.id}: leader gave up ({scored_status}) — marking scored, no weights")
                            store.mark_scored(self._db, spec.id, status=scored_status)
                        else:
                            logger.debug(f"{spec.id}: no scoring results from leader yet")
                        continue

                    ranked = sorted(
                        [ScoringResult(**r) for r in results],
                        key=lambda r: r.final_score,
                        reverse=True,
                    )
                    hotkey_weights = compute_emission_weights(ranked, spec.emission_distribution)

                    logger.info(f"{spec.id}: following leader — weights: "
                                f"{[(k[:8], v) for k, v in hotkey_weights.items() if v > 0]}")
                    store.record_weights(self._db, spec.id, hotkey_weights)
                    store.mark_scored(self._db, spec.id)

            except Exception as e:
                logger.exception(f"Follower loop error: {e}")
            finally:
                await asyncio.sleep(validator_settings.FOLLOWER_POLL_INTERVAL)

    async def run_stage_1(self, spec) -> None:
        from validator.chain_scanner import scan_reveals
        from validator.scorer import dedup_winner
        from competition.scoring import sort_by_self_reported

        logger.info(f"🏆 Stage 1 (reveal/rank) for {spec.id}")

        try:
            reveals = scan_reveals(self.subtensor, spec, self._db)
        except Exception as e:
            self._bump_stage1_or_fail(spec.id, f"scan_reveals failed: {e}")
            return

        logger.debug(f"scan_reveals returned {len(reveals)} reveal(s): {list(reveals.keys())}")
        if not reveals:
            logger.warning(f"{spec.id}: no valid reveals — terminal, no reveals will ever arrive after commit window close")
            store.mark_scored(self._db, spec.id, status="failed_no_reveals")
            return

        registered = set(self.metagraph.hotkeys)
        reveals = {hk: sub for hk, sub in reveals.items() if hk in registered}
        if not reveals:
            logger.warning(f"{spec.id}: no reveals from registered hotkeys — terminal")
            store.mark_scored(self._db, spec.id, status="failed_no_reveals")
            return

        collateral_by_hotkey = {n.hotkey: n.collateral_locked for n in self.metagraph.neurons}
        min_collateral = validator_settings.COLLATERAL_MIN_THRESHOLD
        collateral_failed: dict[str, str] = {}
        if min_collateral > 0:
            def _has_collateral(hotkey: str) -> bool:
                collateral = collateral_by_hotkey.get(hotkey)
                if collateral is None:
                    return True
                return collateral.amount >= min_collateral

            for hk in list(reveals):
                if not _has_collateral(hk):
                    collateral_failed[hk] = f"collateral below threshold ({collateral_by_hotkey[hk]} < {min_collateral})"
                    logger.info(f"{hk[:12]} failed stage 1 — {collateral_failed[hk]}")

        seen_hashes: dict[str, tuple[str, int]] = {}
        dedup_losers: dict[str, tuple[str, int]] = {}
        for hotkey, (submission, block) in sorted(reveals.items(), key=lambda kv: kv[1][1]):
            loser = dedup_winner(seen_hashes, hotkey, submission.file_sha256, block)
            if loser is not None:
                dedup_losers[hotkey] = loser
            else:
                displaced = seen_hashes.get(submission.file_sha256)
                if displaced is not None:
                    dedup_losers[displaced[0]] = (hotkey, block)
                seen_hashes[submission.file_sha256] = (hotkey, block)

        dedup_failed: dict[str, str] = {}
        for hotkey, (winner_hotkey, winner_block) in dedup_losers.items():
            submission, block = reveals[hotkey]
            dedup_failed[hotkey] = (
                f"duplicate file_sha256={submission.file_sha256[:12]} — earlier reveal by "
                f"{winner_hotkey[:12]} at block {winner_block} (this reveal at block {block})"
            )
            logger.info(f"{hotkey[:12]} failed stage 1 — dedup loss")

        eligible = {
            hk: v for hk, v in reveals.items()
            if hk not in collateral_failed and hk not in dedup_failed
        }
        if not eligible:
            logger.warning(f"{spec.id}: no reveals left after collateral/dedup filtering — terminal")
            store.mark_scored(self._db, spec.id, status="failed_no_participants")
            return

        try:
            coordinator = self._get_coordinator()
            available = await asyncio.to_thread(coordinator.list_benchmarks)
            missing = {t.name for t in spec.benchmarks} - available
            if missing:
                self._bump_stage1_or_fail(spec.id, f"coordinator missing benchmarks {missing}")
                return
        except Exception as e:
            self._bump_stage1_or_fail(spec.id, f"cannot reach coordinator: {e}")
            return

        ranked_candidates = sort_by_self_reported({hk: sub for hk, (sub, _block) in eligible.items()}, spec)
        logger.debug(f"Ranked candidates by self-reported score: {[hk[:12] for hk, _ in ranked_candidates]}")

        candidates = [
            {
                "hotkey": hotkey,
                "rank": rank,
                "submission_json": submission.model_dump_json(),
                "reveal_block": reveals[hotkey][1],
                "status": "standby",
            }
            for rank, (hotkey, submission) in enumerate(ranked_candidates)
        ] + [
            {
                "hotkey": hotkey,
                "rank": -1,
                "submission_json": reveals[hotkey][0].model_dump_json(),
                "reveal_block": reveals[hotkey][1],
                "status": "failed",
                "failure_reason": reason,
            }
            for hotkey, reason in {**collateral_failed, **dedup_failed}.items()
        ]

        try:
            store.insert_revealed_candidates(self._db, spec.id, candidates)
        except Exception as e:
            self._bump_stage1_or_fail(spec.id, f"failed to persist revealed_candidates: {e}")
            return

        store.set_stage(self._db, spec.id, "stage2_scoring")
        logger.info(f"{spec.id}: stage 1 done — {len(ranked_candidates)} ranked, "
                    f"{len(collateral_failed) + len(dedup_failed)} failed stage-1 filters")

    def _bump_stage1_or_fail(self, competition_id: str, reason: str) -> None:
        attempts = store.bump_stage1_attempts(self._db, competition_id)
        if attempts >= store.STAGE1_MAX_ATTEMPTS:
            logger.error(f"{competition_id}: stage 1 giving up after {attempts} attempts — {reason}")
            store.mark_scored(self._db, competition_id, status="failed_stage1_infra")
        else:
            logger.warning(f"{competition_id}: stage 1 infra failure, attempt {attempts}/{store.STAGE1_MAX_ATTEMPTS} — {reason}")

    async def run_stage_2(self, spec, current_block: int) -> None:
        from validator.scorer import precheck_one, submit_benchmarks_for_candidate, poll_open_benchmarks
        from competition.precheck_client import PrecheckContainer, is_container_up, stop_container
        from common.models.submission import MinerSubmission

        db = self._db
        cid = spec.id
        done_count = store.count_candidates_by_status(db, cid, ("done",))
        still_active = store.count_candidates_by_status(
            db, cid, ("standby", "prechecking", "queued", "benchmarking")
        )

        if current_block >= spec.scoring_end_block or done_count >= spec.top_n or still_active == 0:
            stop_container(cid)
            self._finalize_stage_3(spec)
            return

        coordinator = self._get_coordinator()

        try:
            await asyncio.to_thread(poll_open_benchmarks, db, cid, spec, coordinator)
        except Exception as e:
            logger.warning(f"{cid}: poll_open_benchmarks error (will retry next tick): {e}")

        concurrency_cap = max(1, validator_settings.SCORING_QUEUE_CONCURRENCY)
        in_flight_benchmarking = store.count_candidates_by_status(db, cid, ("benchmarking",))
        benchmark_slots = concurrency_cap - in_flight_benchmarking
        if benchmark_slots > 0:
            queued = store.candidates_by_status(db, cid, ("queued",))[:benchmark_slots]
            for row in queued:
                submission = MinerSubmission.model_validate_json(row["submission_json"])
                store.set_candidate_status(db, cid, row["hotkey"], "benchmarking")
                err = await asyncio.to_thread(
                    submit_benchmarks_for_candidate, db, cid, row["hotkey"], submission,
                    row["gguf_file"], spec, coordinator,
                )
                if err:
                    logger.warning(f"{row['hotkey'][:12]} {err} — backfilling")
                    store.set_candidate_status(db, cid, row["hotkey"], "failed", err)

        precheck_ctr = PrecheckContainer(competition_id=cid, base_repo=spec.model_repo)
        try:
            await asyncio.to_thread(precheck_ctr.launch)
        except Exception as e:
            logger.warning(f"{cid}: precheck container launch failed (will retry next tick): {e}")
            return

        if not await asyncio.to_thread(is_container_up, cid):
            logger.debug(f"{cid}: precheck container not ready yet, skipping precheck this tick")
            return

        pipeline_count = store.count_candidates_by_status(
            db, cid, ("done", "benchmarking", "queued", "prechecking")
        )
        pipeline_has_room = pipeline_count < spec.top_n + concurrency_cap
        precheck_in_flight = store.count_candidates_by_status(db, cid, ("prechecking",)) > 0
        if pipeline_has_room and not precheck_in_flight:
            standbys = store.candidates_by_status(db, cid, ("standby",))[:1]
            for row in standbys:
                submission = MinerSubmission.model_validate_json(row["submission_json"])
                store.set_candidate_status(db, cid, row["hotkey"], "prechecking")
                try:
                    result = await asyncio.to_thread(precheck_one, row["hotkey"], submission, spec, precheck_ctr, db)
                except Exception as e:
                    logger.warning(f"{row['hotkey'][:12]} precheck raised: {e} — backfilling")
                    store.set_candidate_status(db, cid, row["hotkey"], "failed", f"precheck raised: {e}")
                    continue
                if result.passed:
                    store.mark_precheck_passed(db, cid, row["hotkey"], result.gguf_file, result.measured_memory_kb)
                else:
                    logger.info(f"{row['hotkey'][:12]} failed precheck: {result.reason} — backfilling")
                    store.set_candidate_status(db, cid, row["hotkey"], "failed", result.reason)

    def _finalize_stage_3(self, spec) -> None:
        from competition.scoring import compute_emission_weights
        from common.models.submission import ScoringResult

        cid = spec.id
        results = store.scoring_results_for_competition(self._db, cid)
        ranked = sorted(
            (ScoringResult(hotkey=r["hotkey"], competition_id=cid,
                           final_score=r["final_score"], max_memory_kb=r["max_memory_kb"])
             for r in results),
            key=lambda r: r.final_score, reverse=True,
        )

        hotkey_weights = compute_emission_weights(ranked, spec.emission_distribution)
        logger.debug(f"{cid}: final ranking: {[(r.hotkey[:12], r.final_score) for r in ranked]}")
        logger.debug(f"{cid}: emission weights: {hotkey_weights}")

        store.record_weights(self._db, cid, hotkey_weights)
        store.mark_scored(self._db, cid, status="scored")
        logger.info(f"{cid}: weights computed: {[(k[:8], v) for k, v in hotkey_weights.items() if v > 0]}")

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
                current_block = get_current_block(self.subtensor)

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

        for prefix, task in (("weight_task", self._weight_task if hasattr(self, "_weight_task") else None),
                             ("validator_task", self._validator_task if hasattr(self, "_validator_task") else None)):
            if task:
                status[f"{prefix}_running"] = not task.done()
                status[f"{prefix}_cancelled"] = task.cancelled()
            else:
                status[f"{prefix}_running"] = False

        return status

    def _log_task_status(self, weight_task: asyncio.Task, validator_task: asyncio.Task, task_restart_count: dict):
        """Log the current status of both tasks for debugging."""
        def _state(task: Optional[asyncio.Task]) -> str:
            if not task:
                return "None"
            if task.cancelled():
                return "Cancelled"
            if task.done():
                return "Done/Failed"
            return "Running"

        logger.info(
            f"📊 Task Status - Weight: {_state(weight_task)} (restarts: {task_restart_count['weight_loop']}), "
            f"Validator: {_state(validator_task)} (restarts: {task_restart_count['validator_loop']})"
        )


if __name__ == "__main__":
    gradient_validator = Validator(
        coldkey=validator_settings.WALLET_COLDKEY, wallet_hotkey=validator_settings.WALLET_HOTKEY
    )
    asyncio.run(gradient_validator.run_validator())
