from typing import Dict, List, Tuple
from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import MinerSubmission


def benchmark_composite(scores: Dict[str, float], tasks: List[BenchmarkTask]) -> float:
    """Weighted average of benchmark scores. Tasks missing from scores count as 0."""
    total_weight = sum(t.weight for t in tasks)
    if total_weight == 0:
        return 0.0
    return sum((t.weight / total_weight) * scores.get(t.name, 0.0) for t in tasks)


def efficiency_score(
    scores: Dict[str, float],
    file_size_bytes: int,
    tasks: List[BenchmarkTask],
) -> float:
    """benchmark_composite / file_size_bytes. Higher = more efficient model."""
    if file_size_bytes <= 0:
        return 0.0
    return benchmark_composite(scores, tasks) / file_size_bytes


def passes_floors(
    scores: Dict[str, float],
    tasks: List[BenchmarkTask],
) -> Tuple[bool, List[str]]:
    """Returns (passes, list_of_failed_task_names)."""
    failures = [t.name for t in tasks if scores.get(t.name, 0.0) < t.min_score]
    return len(failures) == 0, failures


def sort_by_self_reported(
    submissions: Dict[str, MinerSubmission],
    spec: CompetitionSpec,
) -> List[Tuple[str, MinerSubmission]]:
    """Sort {hotkey: submission} dict descending by self-reported efficiency score.
    Returns list of (hotkey, submission) tuples."""
    return sorted(
        submissions.items(),
        key=lambda item: efficiency_score(
            item[1].self_reported_scores, item[1].file_size, spec.benchmarks
        ),
        reverse=True,
    )


def compute_emission_weights(
    ranked_results: list,
    distribution: List[float],
) -> Dict[str, float]:
    """
    Map hotkeys to emission weights based on rank.
    ranked_results must be sorted by final_score descending.
    Non-qualifying or disqualified miners receive weight 0.
    """
    weights: Dict[str, float] = {}
    qualifier_idx = 0
    for result in ranked_results:
        if not result.passed_floors or result.disqualified:
            weights[result.hotkey] = 0.0
            continue
        if qualifier_idx < len(distribution):
            weights[result.hotkey] = distribution[qualifier_idx]
            qualifier_idx += 1
        else:
            weights[result.hotkey] = 0.0
    return weights
