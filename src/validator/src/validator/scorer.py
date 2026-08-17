"""
Scorer for TPN — no full model download in the validator process.
"""
import sqlite3
import time
from dataclasses import dataclass
from typing import Dict, Optional

from huggingface_hub import HfApi
from loguru import logger

from common import settings as common_settings
from common.models.competition import CompetitionSpec
from common.models.submission import MinerSubmission
from competition.benchmark_client import Coordinator, RunStatusCode
from competition.model_store import check_repo_public
from competition.precheck_client import PrecheckContainer
from competition.scoring import final_score, passes_floors, passes_memory_cap
from validator import store

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
_hf_api = HfApi()

# Coordinator phases bucketed by expected stall tolerance — see
# BENCHMARK_STARTUP_STALE_SECONDS / BENCHMARK_EXECUTION_STALE_SECONDS.
_QUEUED_PHASES = {"queued", "retry_waiting"}
_STARTUP_PHASES = {
    "requested", "quoted",
    "provisioning", "preparing_worker", "validating_model_server",
    "model_server_validated", "creating_provider_worker",
    "provider_worker_provisioned", "provider_worker_recording",
    "provider_worker_recorded", "local_evaluator_starting",
    "worker_booting",
    "downloading_model", "waiting_for_model_server",
    "hashing_model",
    "preflight", "model_server_ready",
}
_EXECUTION_PHASES = {
    "benchmarking", "lm_eval_running",
    "collecting_results", "lm_eval_finished", "parsing_results",
    "results_ready", "sending_completion_callback",
}

# Buffer applied to the coordinator's own estimated_seconds_remaining before
# using it to extend the stale window — a safety margin on a number the
# coordinator already computed, not an independently tunable knob.
_ESTIMATE_SAFETY_FACTOR = 1.5


def _stale_window(phase: Optional[str]) -> float:
    if phase in _EXECUTION_PHASES:
        return common_settings.BENCHMARK_EXECUTION_STALE_SECONDS
    if phase in _QUEUED_PHASES:
        return common_settings.BENCHMARK_QUEUED_STALE_SECONDS
    return common_settings.BENCHMARK_STARTUP_STALE_SECONDS


def dedup_winner(
    seen: Dict[str, "tuple[str, int]"],
    hotkey: str,
    file_sha256: str,
    reveal_block: int,
) -> Optional["tuple[str, int]"]:
    """
    Cross-hotkey sha256 dedup, keyed on self-reported file_sha256, tiebroken
    by reveal_block (chain-native, unforgeable) then hotkey (deterministic).

    `seen` maps file_sha256 -> (hotkey, reveal_block) of the current in-run
    winner for that hash. Returns None if there's no collision (or this
    candidate is the new winner — caller should update `seen`), or
    (winner_hotkey, winner_block) if this candidate loses.
    """
    existing = seen.get(file_sha256)
    if existing is None:
        return None
    existing_hotkey, existing_block = existing
    if reveal_block < existing_block:
        return None
    if reveal_block == existing_block and hotkey < existing_hotkey:
        return None
    return existing_hotkey, existing_block


@dataclass
class PrecheckResult:
    passed: bool
    reason: str = ""
    gguf_file: str = ""
    measured_memory_kb: int = 0


def precheck_one(
    hotkey: str,
    submission: MinerSubmission,
    spec: CompetitionSpec,
    precheck_ctr: PrecheckContainer,
    conn: sqlite3.Connection,
) -> PrecheckResult:
    if not check_repo_public(submission.repository):
        return PrecheckResult(False, "repo not publicly accessible")

    try:
        files = list(_hf_api.list_repo_files(
            repo_id=submission.repository,
            revision=submission.huggingface_revision,
        ))
    except Exception as e:
        return PrecheckResult(False, f"HF API list_repo_files failed: {e}")

    gguf_file = next((f for f in files if f.endswith(".gguf")), None)
    if not gguf_file:
        return PrecheckResult(False, "no .gguf file found at revision")

    download_url = _HF_RESOLVE.format(
        repo=submission.repository,
        revision=submission.huggingface_revision,
        file=gguf_file,
    )
    logger.debug(f"{hotkey[:12]} starting precheck | url={download_url} | context_length={spec.ram_check_context_length}")
    started = time.monotonic()
    verdict = precheck_ctr.check(download_url, spec.ram_check_context_length)
    logger.debug(f"{hotkey[:12]} precheck call took {time.monotonic() - started:.1f}s")

    if verdict.error:
        logger.warning(f"{hotkey[:12]} precheck error: {verdict.error}")
        return PrecheckResult(False, f"precheck error: {verdict.error}")

    if verdict.provenance and not verdict.provenance.is_derivative:
        notes = "; ".join(verdict.provenance.notes) or "CKA below threshold"
        logger.warning(f"{hotkey[:12]} provenance fail: {notes}")
        return PrecheckResult(False, f"provenance fail: {notes}")

    if verdict.sha256 and verdict.sha256.lower() != submission.file_sha256:
        reason = f"sha256 mismatch: revealed={submission.file_sha256[:12]} actual={verdict.sha256[:12]}"
        store.ban(conn, hotkey, reason)
        logger.warning(f"{hotkey[:12]} BANNED — {reason}")
        return PrecheckResult(False, f"{reason} (hotkey banned)")

    if not verdict.ram:
        return PrecheckResult(False, "no RAM measurement returned by precheck")

    if not verdict.ram.passed:
        return PrecheckResult(False, "llama-cli load failed")

    reported_bytes = submission.max_memory * 1024
    measured_bytes = verdict.ram.ram_bytes
    tolerance = common_settings.RAM_CHECK_LYING_TOLERANCE
    if reported_bytes > 0 and abs(measured_bytes - reported_bytes) / reported_bytes > tolerance:
        diff = abs(measured_bytes - reported_bytes) / reported_bytes
        reason = f"max_memory lie: reported {reported_bytes}B measured {measured_bytes}B ({diff:.1%})"
        logger.warning(f"{hotkey[:12]} {reason}")
        return PrecheckResult(False, reason)

    logger.info(
        f"{hotkey[:12]} precheck OK"
        f" ram={verdict.ram.ram_bytes:,}B"
        + (f" CKA={verdict.provenance.cka:.3f}" if verdict.provenance else "")
    )

    return PrecheckResult(True, gguf_file=gguf_file, measured_memory_kb=measured_bytes // 1024)


def submit_benchmarks_for_candidate(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    submission: MinerSubmission,
    gguf_file: str,
    spec: CompetitionSpec,
    coordinator: Coordinator,
) -> Optional[str]:
    try:
        for task in spec.benchmarks:
            run_id = coordinator.submit(
                submission.repository, submission.huggingface_revision, task.name, [gguf_file],
            )
            store.insert_benchmark_result(
                conn, competition_id, hotkey, task.name,
                submission.repository, submission.huggingface_revision, run_id,
            )
    except Exception as e:
        return f"benchmark submit error: {e}"
    conn.commit()
    return None


_last_log_at: Dict[tuple, Optional[str]] = {}
_last_progress_wall: Dict[tuple, float] = {}


def poll_open_benchmarks(
    conn: sqlite3.Connection,
    competition_id: str,
    spec: CompetitionSpec,
    coordinator: Coordinator,
    current_block: Optional[int] = None,
) -> None:
    rows = store.open_benchmark_results(conn, competition_id)
    now = time.monotonic()
    touched_hotkeys = set()

    for row in rows:
        hotkey, bname = row["hotkey"], row["benchmark_name"]
        key = (competition_id, hotkey, bname)
        touched_hotkeys.add(hotkey)

        try:
            status = coordinator.poll(row["coordinator_run_id"])
        except Exception as e:
            store.update_benchmark_result(conn, competition_id, hotkey, bname, "failed")
            _fail_candidate(conn, competition_id, hotkey, f"poll error: {e}")
            continue

        if status.status == RunStatusCode.COMPLETED:
            score = status.scores.get(bname, 0.0)
            store.update_benchmark_result(conn, competition_id, hotkey, bname, "completed", score=score)
            logger.debug(f"{hotkey[:12]} {bname}={score:.4f}")
            _last_log_at.pop(key, None)
            _last_progress_wall.pop(key, None)
            continue

        if status.status == RunStatusCode.FAILED:
            store.update_benchmark_result(conn, competition_id, hotkey, bname, "failed")
            reason = f"benchmark {bname} failed: {status.failure_reason or 'unknown'}"
            logger.warning(f"{hotkey[:12]} {reason}")
            _fail_candidate(conn, competition_id, hotkey, reason)
            continue

        if status.last_log_at != _last_log_at.get(key):
            _last_log_at[key] = status.last_log_at
            _last_progress_wall[key] = now
        _last_progress_wall.setdefault(key, now)

        store.update_benchmark_result(
            conn, competition_id, hotkey, bname, row["status"],
            phase=status.phase, percent_complete=status.percent_complete, last_message=status.message,
        )

        window = max(
            _stale_window(status.phase),
            (status.estimated_seconds_remaining or 0) * _ESTIMATE_SAFETY_FACTOR,
        )
        if now - _last_progress_wall[key] <= window:
            continue

        window_open = current_block is None or current_block < spec.scoring_end_block
        if window_open:
            logger.warning(
                f"{hotkey[:12]} {bname} stalled (no progress past stale window) — "
                f"scoring window still open, resuming next tick"
            )
            store.update_benchmark_result(conn, competition_id, hotkey, bname, "pending-resume")
        else:
            logger.warning(f"{hotkey[:12]} {bname} stalled, scoring window closed")
            store.update_benchmark_result(conn, competition_id, hotkey, bname, "failed")
            _fail_candidate(conn, competition_id, hotkey, f"benchmark poll timed out (stale progress): {bname}")

    conn.commit()

    for hotkey in touched_hotkeys:
        _resolve_candidate_if_terminal(conn, competition_id, hotkey, spec)


def _fail_candidate(conn: sqlite3.Connection, competition_id: str, hotkey: str, reason: str) -> None:
    store.set_candidate_status(conn, competition_id, hotkey, "failed", reason)


def _resolve_candidate_if_terminal(
    conn: sqlite3.Connection, competition_id: str, hotkey: str, spec: CompetitionSpec,
) -> None:
    candidate = store.get_candidate(conn, competition_id, hotkey)
    if candidate is None or candidate["status"] != "benchmarking":
        return

    rows = store.benchmark_results_for_hotkey(conn, competition_id, hotkey)
    if any(r["status"] not in ("completed", "failed") for r in rows):
        return
    if any(r["status"] == "failed" for r in rows):
        return

    actual_scores = {r["benchmark_name"]: r["score"] for r in rows if r["score"] is not None}
    submission = MinerSubmission.model_validate_json(candidate["submission_json"])

    claimed_scores = submission.self_reported_scores
    tolerance = common_settings.SCORE_LYING_TOLERANCE
    for bname, actual in actual_scores.items():
        claim = claimed_scores.get(bname)
        if claim is None or claim == 0:
            continue
        shortfall = (claim - actual) / claim
        if shortfall > tolerance:
            reason = f"score mismatch on {bname}: claimed={claim:.4f} actual={actual:.4f} ({shortfall:.1%} lower)"
            store.ban(conn, hotkey, reason)
            logger.warning(f"{hotkey[:12]} BANNED — {reason}")
            _fail_candidate(conn, competition_id, hotkey, f"{reason} (hotkey banned)")
            return

    passed, failures = passes_floors(actual_scores, spec.benchmarks)
    if not passed:
        logger.info(f"{hotkey[:12]} floor fail: {failures}")
        _fail_candidate(conn, competition_id, hotkey, f"failed floors: {failures}")
        return

    measured_memory_kb = candidate["measured_memory_kb"] or 0
    if not passes_memory_cap(measured_memory_kb, spec):
        reason = f"exceeded memory cap: {measured_memory_kb}KB > {spec.max_memory_kb}KB"
        logger.info(f"{hotkey[:12]} memory cap fail: {reason}")
        _fail_candidate(conn, competition_id, hotkey, reason)
        return

    score = final_score(actual_scores, measured_memory_kb, spec)
    logger.info(f"{hotkey[:12]} SCORED final={score:.6f} scores={actual_scores}")
    store.record_scoring_result(conn, competition_id, hotkey, score, measured_memory_kb)
    store.set_candidate_status(conn, competition_id, hotkey, "done")
    conn.commit()
