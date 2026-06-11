"""
Stub scorer for TPN.

Verifies model integrity (SHA256, file size, GGUF header) and returns
stub benchmark scores equal to the miner's self-reported scores.

TODO: Replace _stub_scores() with container-based eval when ready.
The container interface will be:
    run_container_eval(gguf_path: str, tasks: List[BenchmarkTask]) -> Dict[str, float]
"""
import os
import tempfile
from typing import Dict
from loguru import logger

from common.models.competition import CompetitionSpec
from common.models.submission import MinerSubmission, ScoringResult
from competition.model_store import (
    check_repo_public,
    download_repo,
    find_gguf_file,
    sha256_file,
    verify_gguf_header,
)
from competition.scoring import efficiency_score, passes_floors
from validator.punishment import check_lying


def score_model(hotkey: str, submission: MinerSubmission, spec: CompetitionSpec) -> ScoringResult:
    logger.info(f"Scoring {hotkey[:12]} | {submission.repository}")

    if not check_repo_public(submission.repository):
        return _disqualify(hotkey, submission, "repo not publicly accessible")

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            local_dir = download_repo(submission.repository, local_dir=tmpdir)
        except Exception as e:
            return _disqualify(hotkey, submission, f"download failed: {e}")

        gguf_path = find_gguf_file(local_dir)
        if not gguf_path:
            return _disqualify(hotkey, submission, "no .gguf file found in repo")

        if not verify_gguf_header(gguf_path):
            return _disqualify(hotkey, submission, "invalid GGUF header")

        actual_sha256 = sha256_file(gguf_path)
        actual_size = os.path.getsize(gguf_path)

        lying = check_lying(
            hotkey=hotkey,
            reveal_size=submission.file_size,
            actual_size=actual_size,
            reveal_sha256=submission.file_sha256,
            actual_sha256=actual_sha256,
            tolerance=spec.score_tolerance,
        )
        if lying:
            return ScoringResult(
                hotkey=hotkey,
                competition_id=submission.competition_id,
                passed_floors=False,
                disqualified=True,
                disqualification_reason="integrity check failed",
                actual_scores={},
                final_score=0.0,
                actual_file_size_bytes=actual_size,
                lying_detected=True,
                eval_backend="stub",
            )

        # STUB: use self-reported scores as actual scores
        actual_scores = _stub_scores(submission, spec)
        logger.info(f"[STUB] {hotkey[:12]} scores: {actual_scores}")

        passed, failures = passes_floors(actual_scores, spec.benchmarks)
        final = efficiency_score(actual_scores, actual_size, spec.benchmarks) if passed else 0.0

        return ScoringResult(
            hotkey=hotkey,
            competition_id=submission.competition_id,
            passed_floors=passed,
            disqualified=False,
            disqualification_reason=f"failed floors: {failures}" if not passed else None,
            actual_scores=actual_scores,
            final_score=final,
            actual_file_size_bytes=actual_size,
            lying_detected=False,
            eval_backend="stub",
        )


def _stub_scores(submission: MinerSubmission, spec: CompetitionSpec) -> Dict[str, float]:
    """
    Returns self-reported scores as actual scores.
    Replace this function with container eval when benchmarking is ready.
    """
    return {
        task.name: min(submission.self_reported_scores.get(task.name, 0.0), 1.0)
        for task in spec.benchmarks
    }


def _disqualify(hotkey: str, submission: MinerSubmission, reason: str) -> ScoringResult:
    logger.warning(f"Disqualified {hotkey[:12]}: {reason}")
    return ScoringResult(
        hotkey=hotkey,
        competition_id=submission.competition_id,
        passed_floors=False,
        disqualified=True,
        disqualification_reason=reason,
        actual_scores={},
        final_score=0.0,
        actual_file_size_bytes=0,
        lying_detected=False,
        eval_backend="stub",
    )
