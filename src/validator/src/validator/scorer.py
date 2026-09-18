"""
Scorer for TPN — no full model download in the validator process.

Miners run their own benchmarks on the coordinator and commit the resulting run
ids on chain. This module verifies those runs really benchmarked the committed
model, then prechecks candidates in verified-score order.
"""
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from huggingface_hub import HfApi
from loguru import logger

from common import settings as common_settings
from common.models.competition import CompetitionSpec
from common.models.submission import MinerSubmission
from competition.benchmark_client import Coordinator, RunStatus, RunStatusCode
from competition.model_store import check_repo_public
from competition.precheck_client import PrecheckContainer
from competition.scoring import final_score, passes_floors, passes_memory_cap
from validator import store

_hf_api = HfApi()


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

    # Match the submitted filename exactly. Picking "the first .gguf" measures
    # RAM on, benchmarks, and hashes a file the miner may not have submitted —
    # and a hash mismatch there bans the hotkey permanently.
    gguf_file = submission.file
    if gguf_file:
        if gguf_file not in files:
            available = ", ".join(f for f in files if f.endswith(".gguf")) or "none"
            return PrecheckResult(
                False,
                f"submitted file '{gguf_file}' not found at revision (.gguf present: {available})",
            )
    else:
        gguf_file = next((f for f in files if f.endswith(".gguf")), None)
        if not gguf_file:
            return PrecheckResult(False, "no .gguf file found at revision")
        logger.warning(f"{hotkey[:12]} submission has no file field — falling back to {gguf_file}")

    logger.debug(
        f"{hotkey[:12]} starting precheck | {submission.repository}"
        f"@{submission.huggingface_revision[:12]}/{gguf_file}"
        f" | context_length={spec.ram_check_context_length}"
    )
    started = time.monotonic()
    verdict = precheck_ctr.check(
        submission.repository,
        submission.huggingface_revision,
        gguf_file,
        spec.ram_check_context_length,
    )
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


# ---------------------------------------------------------------------------
# Run-id verification
#
# A miner-supplied run id proves nothing on its own — the miner could benchmark
# a strong model and commit a weak one. Every run is bound back to the committed
# artifact before its score is accepted.
# ---------------------------------------------------------------------------

@dataclass
class RunVerification:
    benchmark: str
    run_id: str
    ok: bool
    score: float = 0.0
    reason: str = ""
    repo: str = ""
    revision: str = ""


def verify_run(
    submission: MinerSubmission,
    benchmark_name: str,
    run_id: str,
    status: RunStatus,
) -> RunVerification:
    """
    Check a coordinator run really benchmarked the model this submission commits.

    Rejection is never fatal on its own: the benchmark scores 0.0 and the
    competition's floors decide what that is worth.
    """
    fail = lambda reason: RunVerification(
        benchmark=benchmark_name, run_id=run_id, ok=False, reason=reason,
        repo=status.repo or "", revision=status.revision or "",
    )

    if status.status == RunStatusCode.FAILED:
        return fail(f"run failed: {status.failure_reason or 'unknown'}")

    # Hard cutoff: scoring has started, so a run still in flight is out of time.
    if status.status != RunStatusCode.COMPLETED:
        return fail(f"run not complete at scoring time (status={status.status})")

    if not status.repo:
        return fail("coordinator returned no model identity for run")

    if status.repo.lower() != submission.repository.lower():
        return fail(f"repo mismatch: run benchmarked {status.repo}, submission commits {submission.repository}")

    if (status.revision or "").lower() != submission.huggingface_revision.lower():
        return fail(
            f"revision mismatch: run used {(status.revision or '')[:12]}, "
            f"submission commits {submission.huggingface_revision[:12]}"
        )

    # The committed file must be one the run actually loaded. Without this a
    # miner benchmarks one file in the repo and commits another.
    if submission.file not in status.model_files:
        present = ", ".join(status.model_files) or "none"
        return fail(f"committed file '{submission.file}' not among run's model files ({present})")

    # HF only exposes a true content sha256 for LFS files; `xet` and `git_blob`
    # hashes are not comparable to the miner's file_sha256. Repo+revision+path
    # already pins the content, and precheck rehashes the real file, so a
    # non-sha256 algorithm skips this check rather than failing the run.
    algorithm, value = status.file_hashes.get(submission.file, (None, None))
    if algorithm == "sha256" and value and value.lower() != submission.file_sha256:
        return fail(f"file hash mismatch: run has {value[:12]}, submission commits {submission.file_sha256[:12]}")

    if benchmark_name not in status.benchmarks:
        covered = ", ".join(status.benchmarks) or "none"
        return fail(f"run does not cover benchmark '{benchmark_name}' (covers: {covered})")

    # Suite runs report per-benchmark outcomes; a partially completed suite can
    # carry a failed child alongside successful ones.
    item = status.item_status.get(benchmark_name)
    if item and item != "completed":
        return fail(f"benchmark '{benchmark_name}' did not complete in run (status={item})")

    if benchmark_name not in status.scores:
        return fail(f"run returned no score for '{benchmark_name}'")

    return RunVerification(
        benchmark=benchmark_name,
        run_id=run_id,
        ok=True,
        score=status.scores[benchmark_name],
        repo=status.repo or "",
        revision=status.revision or "",
    )


def verify_candidate_runs(
    submission: MinerSubmission,
    spec: CompetitionSpec,
    coordinator: Coordinator,
) -> List[RunVerification]:
    """Verify one run per competition benchmark. Missing runs score 0.0."""
    run_ids = submission.run_ids
    results: List[RunVerification] = []

    for task in spec.benchmarks:
        run_id = run_ids.get(task.name)
        if not run_id:
            results.append(RunVerification(
                benchmark=task.name, run_id="", ok=False,
                reason="no run id submitted for this benchmark",
            ))
            continue
        try:
            status = coordinator.poll(run_id)
        except Exception as e:
            results.append(RunVerification(
                benchmark=task.name, run_id=run_id, ok=False,
                reason=f"coordinator poll failed: {e}",
            ))
            continue
        results.append(verify_run(submission, task.name, run_id, status))

    return results


def verify_all_candidates(
    conn: sqlite3.Connection,
    competition_id: str,
    submissions: Dict[str, MinerSubmission],
    spec: CompetitionSpec,
    coordinator: Coordinator,
    max_workers: int = 5,
) -> Dict[str, Dict[str, float]]:
    """
    Verify every candidate's runs and persist the outcome.

    Returns {hotkey: {benchmark_name: score}} — a rejected or missing run
    contributes 0.0, so every competition benchmark is always present.
    """
    if not submissions:
        return {}

    hotkeys = list(submissions)
    with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
        all_results = list(pool.map(
            lambda hk: verify_candidate_runs(submissions[hk], spec, coordinator),
            hotkeys,
        ))

    verified_scores: Dict[str, Dict[str, float]] = {}
    for hotkey, results in zip(hotkeys, all_results):
        submission = submissions[hotkey]
        for result in results:
            store.record_benchmark_verification(
                conn, competition_id, hotkey, result.benchmark,
                run_id=result.run_id,
                repository=result.repo or submission.repository,
                revision=result.revision or submission.huggingface_revision,
                status="completed" if result.ok else "failed",
                score=result.score,
                message=result.reason,
            )
            if not result.ok:
                logger.info(f"{hotkey[:12]} {result.benchmark} rejected — {result.reason}")

        scores = {r.benchmark: r.score for r in results}
        verified_scores[hotkey] = scores
        logger.debug(f"{hotkey[:12]} verified scores: {scores}")

    return verified_scores


# ---------------------------------------------------------------------------
# Candidate finalization (stage 2, after precheck)
# ---------------------------------------------------------------------------

def finalize_prechecked_candidate(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    spec: CompetitionSpec,
    verified_scores: Dict[str, float],
    measured_memory_kb: int,
) -> bool:
    """
    Apply floors and the memory cap to a candidate that passed precheck, then
    score it. Returns True if the candidate was scored.
    """
    passed, failures = passes_floors(verified_scores, spec.benchmarks)
    if not passed:
        logger.info(f"{hotkey[:12]} floor fail: {failures}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", f"failed floors: {failures}")
        return False

    if not passes_memory_cap(measured_memory_kb, spec):
        reason = f"exceeded memory cap: {measured_memory_kb}KB > {spec.max_memory_kb}KB"
        logger.info(f"{hotkey[:12]} memory cap fail: {reason}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", reason)
        return False

    score = final_score(verified_scores, measured_memory_kb, spec)
    logger.info(f"{hotkey[:12]} SCORED final={score:.6f} scores={verified_scores}")
    store.finalize_candidate(conn, competition_id, hotkey, score, measured_memory_kb)
    return True
