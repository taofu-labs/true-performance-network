"""
Scorer for TPN — nothing is downloaded or run in the validator process.

Miners run their own benchmarks and KL comparison on the coordinator and commit
the resulting run ids on chain. This module verifies those runs really measured
the committed model, and reads every number it scores on out of them: benchmark
scores, the KL fidelity multiplier, and the GPU memory the candidate took to
load.
"""
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger

from common.models.competition import KL_RUN_NAME, CompetitionSpec, CompetitionType
from common.models.submission import MinerSubmission
from competition.benchmark_client import Coordinator, RunStatus, RunStatusCode
from competition.scoring import final_score, kl_norm, passes_floors, passes_memory_cap
from validator import store


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
    # GPU memory the coordinator measured for this run's candidate, in KB. Only
    # the KL run carries it (it loads the committed model against the base);
    # 0 means the run reported none.
    measured_memory_kb: int = 0


def verify_run(
    submission: MinerSubmission,
    benchmark_name: str,
    run_id: str,
    status: RunStatus,
    fluid_epoch_id: Optional[str] = None,
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
    # already pins the content, so a non-sha256 algorithm skips this check rather
    # than failing the run. A mismatch fails the run; nothing bans a hotkey.
    algorithm, value = status.file_hashes.get(submission.file, (None, None))
    if algorithm == "sha256" and value and value.lower() != submission.file_sha256:
        return fail(f"file hash mismatch: run has {value[:12]}, submission commits {submission.file_sha256[:12]}")

    # A winner's model has to be downloadable by anyone. The validator never
    # contacts the Hub, so the run's own view of the repo is the evidence —
    # a miner who publishes only long enough to benchmark is caught here.
    if status.repo_private or status.repo_gated:
        state = "private" if status.repo_private else "gated"
        return fail(f"repo was {state} when the run read it — publish it before scoring")

    if benchmark_name not in status.benchmarks:
        covered = ", ".join(status.benchmarks) or "none"
        return fail(f"run does not cover benchmark '{benchmark_name}' (covers: {covered})")

    # Suite runs report per-benchmark outcomes; a partially completed suite can
    # carry a failed child alongside successful ones.
    item = status.item_status.get(benchmark_name)
    if item and item != "completed":
        return fail(f"benchmark '{benchmark_name}' did not complete in run (status={item})")

    if fluid_epoch_id and status.fluid_epoch_id != fluid_epoch_id:
        return fail(f"run used Fluid epoch {status.fluid_epoch_id or 'none'}, competition requires {fluid_epoch_id}")

    if benchmark_name not in status.scores:
        return fail(f"run returned no score for '{benchmark_name}'")

    return RunVerification(
        benchmark=benchmark_name,
        run_id=run_id,
        ok=True,
        score=status.scores[benchmark_name],
        repo=status.repo or "",
        revision=status.revision or "",
        # Suite-level: one model load serves every benchmark in the run. Absent
        # on older runs, which is why the KL run is the authoritative source.
        measured_memory_kb=(status.measured_memory_bytes or 0) // 1024,
    )


# `control_to_candidate` compares against an F16 control converted from the same
# base on the same pod, so both sides run llama.cpp and engine noise cancels,
# leaving quantization damage alone. `original_to_candidate` conflates the two:
# measured control noise is 0.004 nats at P99. P99 rather than the mean prices
# tail damage: a Q4_K_M means 0.025 nats but reaches 0.146 at P99.
KL_PAIRING = "control_to_candidate"
KL_QUANTILE = "0.99"


def verify_kl_run(
    submission: MinerSubmission,
    spec: CompetitionSpec,
    run_id: str,
    status: RunStatus,
) -> RunVerification:
    """
    Check a comparison run measured the committed model against the
    competition's base on the competition's platform corpus, and return its KL
    fidelity multiplier as `score`.

    Rejection is never fatal and never bans: the multiplier is 0.0, which zeroes
    the final score through the multiplication without removing the candidate.
    An unverifiable run must never read as perfect fidelity.
    """
    fail = lambda reason: RunVerification(
        benchmark=KL_RUN_NAME, run_id=run_id, ok=False, reason=reason,
        repo=status.repo or "", revision=status.revision or "",
    )

    if status.status == RunStatusCode.FAILED:
        return fail(f"KL run failed: {status.failure_reason or 'unknown'}")

    if status.status != RunStatusCode.COMPLETED:
        return fail(f"KL run not complete at scoring time (status={status.status})")

    comparison = status.comparison
    if not isinstance(comparison, dict):
        return fail("run carries no comparison result — not a comparison run")

    # A verifier run reports one 0-100 score over a private sharded suite, not
    # pairings over the competition's corpus.
    if comparison.get("protocol") != "fidelity-v1":
        return fail(f"wrong comparison protocol {comparison.get('protocol')!r}, need fidelity-v1")

    if comparison.get("outcome") != "complete":
        return fail(f"comparison outcome {comparison.get('outcome')!r}, need complete")

    # A gguf reference never loads the HF base: no base-side pairing, nothing to
    # pin model_repo against.
    if comparison.get("reference") != "transformers":
        return fail(f"comparison reference {comparison.get('reference')!r}, need transformers")

    # Never optional: an unpinned base lets the miner choose what they are
    # measured against. A spec stored before that rule reached CompetitionSpec
    # fails here rather than scoring a free multiplier.
    if not spec.model_repo:
        return fail("competition sets kl_corpus_id but pins no model_repo — cannot verify the base")
    if not status.base_repo:
        return fail("coordinator returned no base model identity for KL run")
    if status.base_repo.lower() != spec.model_repo.lower():
        return fail(
            f"base mismatch: KL run compared against {status.base_repo}, "
            f"competition requires {spec.model_repo}"
        )

    # A miner-uploaded dataset is text they chose.
    dataset = status.dataset or {}
    if dataset.get("kind") != "platform":
        return fail(f"KL run used a {dataset.get('kind')!r} dataset, need the platform corpus")
    if dataset.get("corpus_id") != spec.kl_corpus_id:
        return fail(
            f"corpus mismatch: KL run used {dataset.get('corpus_id')!r}, "
            f"competition requires {spec.kl_corpus_id!r}"
        )

    # Same binding as a benchmark run: without it a miner submits a comparison
    # of some other, better-behaved model.
    if not status.repo:
        return fail("coordinator returned no candidate model identity for KL run")
    if status.repo.lower() != submission.repository.lower():
        return fail(f"repo mismatch: KL run compared {status.repo}, submission commits {submission.repository}")
    if (status.revision or "").lower() != submission.huggingface_revision.lower():
        return fail(
            f"revision mismatch: KL run used {(status.revision or '')[:12]}, "
            f"submission commits {submission.huggingface_revision[:12]}"
        )
    if submission.file not in status.model_files:
        present = ", ".join(status.model_files) or "none"
        return fail(f"committed file '{submission.file}' not among KL run's model files ({present})")

    algorithm, value = status.file_hashes.get(submission.file, (None, None))
    if algorithm == "sha256" and value and value.lower() != submission.file_sha256:
        return fail(f"file hash mismatch: KL run has {value[:12]}, submission commits {submission.file_sha256[:12]}")

    if status.repo_private or status.repo_gated:
        state = "private" if status.repo_private else "gated"
        return fail(f"repo was {state} when the KL run read it — publish it before scoring")

    pairing = (comparison.get("pairings") or {}).get(KL_PAIRING)
    if not isinstance(pairing, dict):
        return fail(f"comparison has no '{KL_PAIRING}' pairing")

    # The DTO filters `quantiles` on presence, so the key can be absent — that is
    # unverifiable, not zero divergence.
    quantile = ((pairing.get("kl") or {}).get("quantiles") or {}).get(KL_QUANTILE)
    if not isinstance(quantile, (int, float)) or isinstance(quantile, bool):
        return fail(f"comparison has no P{KL_QUANTILE} KL for '{KL_PAIRING}'")

    # The competition's memory cap is enforced against this figure, so a run
    # that reports none cannot be scored — the cap would silently pass.
    measured_bytes = status.measured_memory_bytes
    if not measured_bytes or measured_bytes <= 0:
        return fail("KL run reported no measured GPU memory for the candidate")

    multiplier = kl_norm(float(quantile))
    return RunVerification(
        benchmark=KL_RUN_NAME,
        run_id=run_id,
        ok=True,
        score=multiplier,
        reason=(
            f"KL P{KL_QUANTILE}={quantile:.6g} nats on {KL_PAIRING} -> x{multiplier:.4f}"
            f", {measured_bytes // 1024:,} KB GPU"
        ),
        repo=status.repo or "",
        revision=status.revision or "",
        measured_memory_kb=measured_bytes // 1024,
    )


def verify_candidate_runs(
    submission: MinerSubmission,
    spec: CompetitionSpec,
    coordinator: Coordinator,
) -> List[RunVerification]:
    """
    Verify one run per competition benchmark, plus the KL comparison run when
    the competition configures one. Missing runs score 0.0.
    """
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
        results.append(verify_run(submission, task.name, run_id, status, task.fluid_epoch_id))

    if spec.kl_corpus_id:
        kl_run_id = run_ids.get(KL_RUN_NAME)
        if not kl_run_id:
            results.append(RunVerification(
                benchmark=KL_RUN_NAME, run_id="", ok=False,
                reason="no KL comparison run id submitted",
            ))
        else:
            try:
                results.append(verify_kl_run(submission, spec, kl_run_id, coordinator.poll(kl_run_id)))
            except Exception as e:
                results.append(RunVerification(
                    benchmark=KL_RUN_NAME, run_id=kl_run_id, ok=False,
                    reason=f"coordinator poll failed: {e}",
                ))

    return results


@dataclass
class CandidateVerification:
    """Everything scoring needs about one candidate, all of it from the coordinator."""
    scores: Dict[str, float] = field(default_factory=dict)
    # GPU memory measured for the committed model, in KB. 0 when no verified run
    # reported one, which fails the memory cap rather than passing it silently.
    measured_memory_kb: int = 0


def measured_memory_kb(results: List[RunVerification]) -> int:
    """
    The GPU memory figure to score on: the largest any *verified* run reported.

    The KL run is the authoritative one — it loads the committed model against
    the competition's base. Benchmark runs carry a suite-level figure too, so a
    competition without a KL check still gets a number. Taking the maximum means
    a candidate is held to its heaviest observed load, never its lightest.

    Rejected runs are ignored: their numbers describe some other model.
    """
    return max((r.measured_memory_kb for r in results if r.ok), default=0)


def verify_all_candidates(
    conn: sqlite3.Connection,
    competition_id: str,
    submissions: Dict[str, MinerSubmission],
    spec: CompetitionSpec,
    coordinator: Coordinator,
    max_workers: int = 5,
) -> Dict[str, CandidateVerification]:
    """
    Verify every candidate's runs and persist the outcome.

    Returns {hotkey: CandidateVerification} — a rejected or missing run
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

    verified: Dict[str, CandidateVerification] = {}
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
        memory_kb = measured_memory_kb(results)
        verified[hotkey] = CandidateVerification(scores=scores, measured_memory_kb=memory_kb)
        logger.debug(f"{hotkey[:12]} verified scores: {scores} memory={memory_kb:,}KB")

    return verified


# ---------------------------------------------------------------------------
# Candidate finalization
# ---------------------------------------------------------------------------

def finalize_candidate(
    conn: sqlite3.Connection,
    competition_id: str,
    hotkey: str,
    spec: CompetitionSpec,
    verification: CandidateVerification,
) -> bool:
    """
    Apply the competition's floors and memory cap to a verified candidate, then
    score it. Returns True if the candidate was scored.

    Every input comes from verified coordinator runs, so this does no I/O beyond
    the database.
    """
    scores = verification.scores
    memory_kb = verification.measured_memory_kb

    passed, failures = passes_floors(scores, spec.benchmarks)
    if not passed:
        logger.info(f"{hotkey[:12]} floor fail: {failures}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", f"failed floors: {failures}")
        return False

    # No figure means no run reported one. Treating that as 0 would pass any cap,
    # so it fails instead.
    if not memory_kb:
        reason = "no verified run reported the candidate's GPU memory"
        logger.info(f"{hotkey[:12]} {reason}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", reason)
        return False

    if not passes_memory_cap(memory_kb, spec):
        reason = f"exceeded memory cap: {memory_kb}KB > {spec.max_gpu_memory_kb}KB"
        logger.info(f"{hotkey[:12]} memory cap fail: {reason}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", reason)
        return False

    score = final_score(scores, memory_kb, spec)

    # A zeroed score means something required was missing or unverifiable — most
    # often the KL run, whose multiplier is 0.0 when it cannot be verified. Such
    # a candidate must not occupy a payable rank: emission shares go by position,
    # so in a thin field a zero would still draw a share.
    #
    # Only the multiplicative branch can reach 0 this way; BENCHMARK_FLOOR scores
    # negated memory, where 0 is unreachable (memory_kb > 0 is checked above).
    if spec.competition_type == CompetitionType.GPU_MEMORY_CEILING and score <= 0.0:
        reason = f"scored 0 — nothing payable (scores={scores})"
        logger.info(f"{hotkey[:12]} {reason}")
        store.set_candidate_status(conn, competition_id, hotkey, "failed", reason)
        return False

    logger.info(f"{hotkey[:12]} SCORED final={score:.6f} scores={scores} memory={memory_kb:,}KB")
    store.finalize_candidate(conn, competition_id, hotkey, score, memory_kb)
    return True
