"""
Scorer for TPN — no full model download in the validator process.

Flow per scoring run:
  1. precheck_one()   — HF metadata + container (provenance + RAM). Sync, sequential.
  2. benchmark_one()  — submit all benchmarks, poll until done. Async, runs in parallel
                        across miners (coordinator manages its own queue).
"""
import asyncio
import sqlite3
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, List, Optional, Union

from huggingface_hub import HfApi
from loguru import logger

from common import settings as common_settings
from common.models.competition import CompetitionSpec
from common.models.submission import MinerSubmission, ScoringResult
from competition.benchmark_client import Coordinator, RunStatusCode
from competition.model_store import check_repo_public
from competition.precheck_client import PrecheckContainer
from competition.scoring import final_score, passes_floors, passes_memory_cap
from validator import store

_HF_RESOLVE = "https://huggingface.co/{repo}/resolve/{revision}/{file}"
_hf_api = HfApi()


# ---------------------------------------------------------------------------
# Outcome
# ---------------------------------------------------------------------------

class OutcomeKind(Enum):
    SCORED = auto()
    SKIPPED = auto()       # any non-lie failure → backfill
    DISQUALIFIED = auto()  # max_memory lie → also backfill, but flagged


@dataclass
class ScoringOutcome:
    kind: OutcomeKind
    result: ScoringResult
    reason: str = ""


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
class PrecheckPass:
    """Carries everything benchmark_one needs — produced by precheck_one on success."""
    hotkey: str
    submission: MinerSubmission
    gguf_file: str
    measured_memory_kb: int


# ---------------------------------------------------------------------------
# Phase 1: precheck (sync — runs sequentially, one at a time)
# ---------------------------------------------------------------------------

def precheck_one(
    hotkey: str,
    submission: MinerSubmission,
    spec: CompetitionSpec,
    precheck_ctr: Optional[PrecheckContainer],
    conn: sqlite3.Connection,
) -> Union[PrecheckPass, ScoringOutcome]:
    """
    Run HF metadata check + container precheck.
    Returns PrecheckPass on success, ScoringOutcome on skip/disqualify.
    Blocking — call via asyncio.to_thread.
    """
    if precheck_ctr is None:
        return _skip(hotkey, submission, "precheck container unavailable")

    if not check_repo_public(submission.repository):
        return _skip(hotkey, submission, "repo not publicly accessible")

    try:
        files = list(_hf_api.list_repo_files(
            repo_id=submission.repository,
            revision=submission.huggingface_revision,
        ))
    except Exception as e:
        return _skip(hotkey, submission, f"HF API list_repo_files failed: {e}")

    gguf_file = next((f for f in files if f.endswith(".gguf")), None)
    if not gguf_file:
        return _skip(hotkey, submission, "no .gguf file found at revision")

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
        return _skip(hotkey, submission, f"precheck error: {verdict.error}")

    if verdict.provenance and not verdict.provenance.is_derivative:
        notes = "; ".join(verdict.provenance.notes) or "CKA below threshold"
        logger.warning(f"{hotkey[:12]} provenance fail: {notes}")
        return _skip(hotkey, submission, f"provenance fail: {notes}")

    if verdict.sha256 and verdict.sha256.lower() != submission.file_sha256:
        reason = f"sha256 mismatch: revealed={submission.file_sha256[:12]} actual={verdict.sha256[:12]}"
        store.ban(conn, hotkey, reason)
        logger.warning(f"{hotkey[:12]} BANNED — {reason}")
        return _skip(hotkey, submission, f"{reason} (hotkey banned)")

    if not verdict.ram:
        return _skip(hotkey, submission, "no RAM measurement returned by precheck")

    if not verdict.ram.passed:
        return _skip(hotkey, submission, "llama-cli load failed")

    reported_bytes = submission.max_memory * 1024
    measured_bytes = verdict.ram.ram_bytes
    tolerance = getattr(common_settings, "RAM_CHECK_LYING_TOLERANCE", 0.01)
    if reported_bytes > 0 and abs(measured_bytes - reported_bytes) / reported_bytes > tolerance:
        diff = abs(measured_bytes - reported_bytes) / reported_bytes
        logger.warning(f"{hotkey[:12]} max_memory lie: reported={reported_bytes} measured={measured_bytes} diff={diff:.1%}")
        return _disqualify(
            hotkey, submission,
            f"max_memory lie: reported {reported_bytes}B measured {measured_bytes}B ({diff:.1%})",
        )

    logger.info(
        f"{hotkey[:12]} precheck OK"
        f" ram={verdict.ram.ram_bytes:,}B"
        + (f" CKA={verdict.provenance.cka:.3f}" if verdict.provenance else "")
    )

    return PrecheckPass(
        hotkey=hotkey, submission=submission, gguf_file=gguf_file,
        measured_memory_kb=measured_bytes // 1024,
    )


# ---------------------------------------------------------------------------
# Phase 2: benchmark (async — all miners run in parallel)
# ---------------------------------------------------------------------------

async def benchmark_one(
    p: PrecheckPass,
    spec: CompetitionSpec,
    coordinator: Coordinator,
    conn: sqlite3.Connection,
    poll_interval: float = 30.0,
    max_wait_seconds: float = None,
    current_block: Optional[int] = None,
    run_ids: Optional[Dict[str, str]] = None,
    submit_semaphore: Optional[asyncio.Semaphore] = None,
) -> ScoringOutcome:
    """
    Submit all benchmarks for one miner and poll until done.

    `run_ids`, if given, are coordinator run_ids already submitted and
    persisted in a prior process (resumed after an autoupdater restart) —
    submit is skipped for any benchmark name already present.

    A validator restart mid-poll is common (autoupdater can SIGKILL every
    ~15min while a scoring window stays open for hours-days), so run_ids are
    persisted to `benchmark_runs` immediately after each submit, and a poll
    timeout only becomes a hard skip once the competition's scoring window
    (spec.scoring_end_block) has actually closed — otherwise the run is left
    'pending-resume' for the next tick's reconciliation pass to pick back up.

    `submit_semaphore`, if given, bounds concurrent *submits* only across all
    benchmark_one() calls running in the same asyncio.gather — submit and
    poll have different coordinator capacity profiles, so polling stays
    unbounded/parallel once a run_id is accepted.
    """
    hotkey, submission = p.hotkey, p.submission
    if max_wait_seconds is None:
        max_wait_seconds = getattr(common_settings, "BENCHMARK_POLL_TIMEOUT_SECONDS", 5400)

    run_ids = dict(run_ids or {})
    try:
        for task in spec.benchmarks:
            if task.name in run_ids:
                continue
            if submit_semaphore is not None:
                async with submit_semaphore:
                    run_id = await asyncio.to_thread(
                        coordinator.submit,
                        submission.repository,
                        submission.huggingface_revision,
                        task.name,
                        [p.gguf_file],
                    )
            else:
                run_id = await asyncio.to_thread(
                    coordinator.submit,
                    submission.repository,
                    submission.huggingface_revision,
                    task.name,
                    [p.gguf_file],
                )
            run_ids[task.name] = run_id
            store.upsert_benchmark_run(
                conn, submission.competition_id, hotkey, task.name,
                submission.repository, submission.huggingface_revision, run_id,
            )
    except Exception as e:
        return _skip(hotkey, submission, f"benchmark submit error: {e}")

    # Poll
    actual_scores: Dict[str, float] = {}
    pending = dict(run_ids)
    deadline = time.monotonic() + max_wait_seconds
    while pending:
        if time.monotonic() >= deadline:
            window_open = current_block is None or current_block < spec.scoring_end_block
            if window_open:
                logger.warning(
                    f"{hotkey[:12]} benchmark poll timeout after {max_wait_seconds:.0f}s, "
                    f"still pending: {list(pending)} — scoring window still open, resuming next tick"
                )
                for bname in pending:
                    store.set_benchmark_run_status(
                        conn, submission.competition_id, hotkey, bname, "pending-resume"
                    )
                return _skip(
                    hotkey, submission,
                    f"benchmark poll pending-resume, pending: {list(pending)}",
                    actual_scores=actual_scores,
                )
            logger.warning(f"{hotkey[:12]} benchmark poll timeout after {max_wait_seconds:.0f}s, scoring window closed: {list(pending)}")
            return _skip(
                hotkey, submission,
                f"benchmark poll timeout after {max_wait_seconds:.0f}s, pending: {list(pending)}",
                actual_scores=actual_scores,
            )
        await asyncio.sleep(poll_interval)
        for bname in list(pending):
            try:
                status = await asyncio.to_thread(coordinator.poll, pending[bname])
            except Exception as e:
                return _skip(hotkey, submission, f"poll error: {e}")
            if status.status == RunStatusCode.COMPLETED:
                actual_scores[bname] = status.scores.get(bname, 0.0)
                del pending[bname]
                store.set_benchmark_run_status(conn, submission.competition_id, hotkey, bname, "completed")
                logger.debug(f"{hotkey[:12]} {bname}={actual_scores[bname]:.4f}")
            elif status.status == RunStatusCode.FAILED:
                store.set_benchmark_run_status(conn, submission.competition_id, hotkey, bname, "failed")
                return _skip(hotkey, submission, f"benchmark {bname} failed: {status.failure_reason or 'unknown'}")

    # Score-mismatch check: miner claimed a benchmark score in its reveal
    # higher than what the coordinator actually measured. Only under-delivery
    # is a lie worth banning — actual scoring higher than claimed is not
    # penalized (e.g. non-determinism, claim rounded down).
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
            return _skip(hotkey, submission, f"{reason} (hotkey banned)", actual_scores=actual_scores)

    # Floor check
    passed, failures = passes_floors(actual_scores, spec.benchmarks)
    if not passed:
        logger.info(f"{hotkey[:12]} floor fail: {failures}")
        return _skip(hotkey, submission, f"failed floors: {failures}", actual_scores=actual_scores)

    # Memory cap check (ram_ceiling competitions only)
    if not passes_memory_cap(submission.max_memory, spec):
        logger.info(f"{hotkey[:12]} memory cap fail: {submission.max_memory}KB > {spec.max_memory_kb}KB")
        return _skip(
            hotkey, submission,
            f"exceeded memory cap: {submission.max_memory}KB > {spec.max_memory_kb}KB",
            actual_scores=actual_scores,
        )

    final = final_score(actual_scores, p.measured_memory_kb, spec)
    logger.info(f"{hotkey[:12]} SCORED final={final:.6f} scores={actual_scores}")
    return ScoringOutcome(
        kind=OutcomeKind.SCORED,
        result=ScoringResult(
            hotkey=hotkey,
            competition_id=submission.competition_id,
            passed_floors=True,
            disqualified=False,
            actual_scores=actual_scores,
            final_score=final,
            max_memory_kb=p.measured_memory_kb,
            lying_detected=False,
            eval_backend="coordinator",
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _skip(
    hotkey: str,
    submission: MinerSubmission,
    reason: str,
    actual_scores: Optional[Dict[str, float]] = None,
) -> ScoringOutcome:
    return ScoringOutcome(
        kind=OutcomeKind.SKIPPED,
        reason=reason,
        result=ScoringResult(
            hotkey=hotkey,
            competition_id=submission.competition_id,
            passed_floors=False,
            disqualified=False,
            disqualification_reason=reason,
            actual_scores=actual_scores or {},
            final_score=0.0,
            max_memory_kb=submission.max_memory,
            lying_detected=False,
            eval_backend="coordinator",
        ),
    )


def _disqualify(hotkey: str, submission: MinerSubmission, reason: str) -> ScoringOutcome:
    logger.warning(f"DISQUALIFIED {hotkey[:12]}: {reason}")
    return ScoringOutcome(
        kind=OutcomeKind.DISQUALIFIED,
        reason=reason,
        result=ScoringResult(
            hotkey=hotkey,
            competition_id=submission.competition_id,
            passed_floors=False,
            disqualified=True,
            disqualification_reason=reason,
            actual_scores={},
            final_score=0.0,
            max_memory_kb=submission.max_memory,
            lying_detected=True,
            eval_backend="coordinator",
        ),
    )

