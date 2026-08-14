from typing import Dict, List, Tuple
from common.models.competition import BenchmarkTask, CompetitionSpec, CompetitionType
from common.models.submission import MinerSubmission


def benchmark_composite(scores: Dict[str, float], tasks: List[BenchmarkTask]) -> float:
    """Weighted average of benchmark scores. Tasks missing from scores count as 0."""
    total_weight = sum(t.weight for t in tasks)
    if total_weight == 0:
        return 0.0
    return sum((t.weight / total_weight) * scores.get(t.name, 0.0) for t in tasks)


def final_score(
    scores: Dict[str, float],
    max_memory_kb: int,
    spec: CompetitionSpec,
) -> float:
    """
    Score a submission per the competition's type. Higher is always better
    (rank descending), so benchmark_floor scores are negated memory.

    - BENCHMARK_FLOOR: rank by lowest max_memory_kb (benchmarks are pass/fail floors).
    - RAM_CEILING: rank by highest benchmark composite (max_memory_kb is a pass/fail cap).
    """
    if spec.competition_type == CompetitionType.RAM_CEILING:
        return benchmark_composite(scores, spec.benchmarks)
    return -float(max_memory_kb)


def passes_floors(
    scores: Dict[str, float],
    tasks: List[BenchmarkTask],
) -> Tuple[bool, List[str]]:
    """Returns (passes, list_of_failed_task_names)."""
    failures = [t.name for t in tasks if scores.get(t.name, 0.0) < t.min_score]
    return len(failures) == 0, failures


def passes_memory_cap(max_memory_kb: int, spec: CompetitionSpec) -> bool:
    """RAM_CEILING only: max_memory_kb must not exceed spec.max_memory_kb."""
    if spec.competition_type != CompetitionType.RAM_CEILING:
        return True
    return max_memory_kb <= spec.max_memory_kb


def sort_by_self_reported(
    submissions: Dict[str, MinerSubmission],
    spec: CompetitionSpec,
) -> List[Tuple[str, MinerSubmission]]:
    """Sort {hotkey: submission} dict descending by self-reported final score
    (lowest claimed memory first for benchmark_floor, highest claimed benchmark for
    ram_ceiling). Ties broken by benchmark composite. Returns list of
    (hotkey, submission) tuples."""
    return sorted(
        submissions.items(),
        key=lambda item: (
            final_score(item[1].self_reported_scores, item[1].max_memory, spec),
            benchmark_composite(item[1].self_reported_scores, spec.benchmarks),
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
    """
    weights: Dict[str, float] = {}
    for idx, result in enumerate(ranked_results):
        weights[result.hotkey] = distribution[idx] if idx < len(distribution) else 0.0
    return weights


def aggregate_competition_weights(
    distributing: List[Tuple[CompetitionSpec, Dict[str, float]]],
    registered_hotkeys: set,
) -> Dict[str, float]:
    """
    Sum per-hotkey weight across every currently-distributing competition,
    scaled by each competition's emission_weight share of the total pool.

    distributing: (spec, hotkey_shares) pairs — hotkey_shares is the output
      of compute_emission_weights for that competition (fractions of that
      competition's own pool, summing to <= 1.0).
    registered_hotkeys: metagraph hotkeys. Winners not in this set are
      dropped — their share is not redistributed to anyone else.

    Returns hotkey -> weight in [0, 1], summed across competitions (a hotkey
    winning in multiple competitions gets the sum). If the summed total
    exceeds 1.0, every value is scaled down by 1/total. Never scales up.
    """
    totals: Dict[str, float] = {}
    for spec, hotkey_shares in distributing:
        for hotkey, share in hotkey_shares.items():
            if share <= 0 or hotkey not in registered_hotkeys:
                continue
            totals[hotkey] = totals.get(hotkey, 0.0) + spec.emission_weight * share

    grand_total = sum(totals.values())
    if grand_total > 1.0:
        totals = {hk: w / grand_total for hk, w in totals.items()}
    return totals
