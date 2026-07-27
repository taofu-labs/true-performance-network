import pytest

from common.models.competition import BenchmarkTask, CompetitionSpec, CompetitionType
from common.models.submission import Claim, MinerSubmission, ScoringResult
from competition.scoring import (
    aggregate_competition_weights,
    benchmark_composite,
    compute_emission_weights,
    final_score,
    passes_floors,
    passes_memory_cap,
    sort_by_self_reported,
)


def make_spec(id: str, emission_weight: float, distribution_blocks: int = 100) -> CompetitionSpec:
    return CompetitionSpec(
        id=id,
        name=id,
        start_block=0,
        commit_end_block=10,
        scoring_end_block=20,
        emission_distribution=[1.0],
        top_n=1,
        emission_weight=emission_weight,
        distribution_blocks=distribution_blocks,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
    )


def test_two_competitions_disjoint_winners_sum_to_full_share():
    spec_a = make_spec("a", 0.5)
    spec_b = make_spec("b", 0.5)
    result = aggregate_competition_weights(
        [(spec_a, {"hk1": 1.0}), (spec_b, {"hk2": 1.0})],
        registered_hotkeys={"hk1", "hk2"},
    )
    assert result == {"hk1": 0.5, "hk2": 0.5}
    assert sum(result.values()) == 1.0


def test_hotkey_winning_multiple_competitions_sums():
    spec_a = make_spec("a", 0.5)
    spec_b = make_spec("b", 0.5)
    result = aggregate_competition_weights(
        [(spec_a, {"hk1": 1.0}), (spec_b, {"hk1": 1.0})],
        registered_hotkeys={"hk1"},
    )
    assert result == {"hk1": 1.0}


def test_unregistered_hotkey_dropped_not_redistributed():
    spec_a = make_spec("a", 0.5)
    result = aggregate_competition_weights(
        [(spec_a, {"hk1": 1.0})],
        registered_hotkeys=set(),
    )
    assert result == {}


def test_over_100_percent_normalizes_down():
    specs = [make_spec(f"c{i}", 0.5) for i in range(3)]
    shares = [{f"hk{i}": 1.0} for i in range(3)]
    result = aggregate_competition_weights(
        list(zip(specs, shares)),
        registered_hotkeys={"hk0", "hk1", "hk2"},
    )
    assert abs(sum(result.values()) - 1.0) < 1e-9
    for w in result.values():
        assert abs(w - (0.5 / 1.5)) < 1e-9


def test_nothing_distributing_returns_empty():
    assert aggregate_competition_weights([], registered_hotkeys={"hk1"}) == {}


def test_is_distributing_and_active_boundaries():
    spec = make_spec("a", 0.5, distribution_blocks=100)
    assert spec.distribution_end_block() == 120
    assert spec.is_distributing(20) is True
    assert spec.is_distributing(119) is True
    assert spec.is_distributing(120) is False
    assert spec.is_active(119) is True
    assert spec.is_active(120) is False


def make_benchmark_floor_spec(**overrides) -> CompetitionSpec:
    defaults = dict(
        id="rf",
        name="rf",
        start_block=0,
        commit_end_block=10,
        scoring_end_block=20,
        emission_distribution=[0.6, 0.4],
        top_n=2,
        emission_weight=1.0,
        benchmarks=[
            BenchmarkTask(name="mmlu", min_score=0.5, weight=0.5),
            BenchmarkTask(name="gsm8k", min_score=0.3, weight=0.5),
        ],
    )
    defaults.update(overrides)
    return CompetitionSpec(**defaults)


def make_ram_ceiling_spec(**overrides) -> CompetitionSpec:
    defaults = dict(
        id="bc",
        name="bc",
        start_block=0,
        commit_end_block=10,
        scoring_end_block=20,
        emission_distribution=[1.0],
        top_n=1,
        emission_weight=1.0,
        competition_type=CompetitionType.RAM_CEILING,
        max_memory_kb=1000,
        benchmarks=[
            BenchmarkTask(name="mmlu", min_score=0.5, weight=0.5),
            BenchmarkTask(name="gsm8k", min_score=0.3, weight=0.5),
        ],
    )
    defaults.update(overrides)
    return CompetitionSpec(**defaults)


def test_benchmark_composite_weighted_average_missing_counts_zero():
    tasks = [
        BenchmarkTask(name="mmlu", min_score=0.0, weight=0.75),
        BenchmarkTask(name="gsm8k", min_score=0.0, weight=0.25),
    ]
    assert benchmark_composite({"mmlu": 0.8}, tasks) == pytest.approx(0.6)  # gsm8k missing -> 0


def test_benchmark_composite_zero_total_weight_returns_zero():
    tasks = [BenchmarkTask(name="mmlu", min_score=0.0, weight=0.0)]
    assert benchmark_composite({"mmlu": 0.9}, tasks) == 0.0


def test_final_score_benchmark_floor_is_negated_memory():
    spec = make_benchmark_floor_spec()
    assert final_score({}, 5000, spec) == -5000.0


def test_final_score_ram_ceiling_is_composite():
    spec = make_ram_ceiling_spec()
    assert final_score({"mmlu": 0.8, "gsm8k": 0.4}, 500, spec) == pytest.approx(0.6)


def test_passes_floors_all_pass_and_one_fail():
    tasks = [
        BenchmarkTask(name="mmlu", min_score=0.5, weight=0.5),
        BenchmarkTask(name="gsm8k", min_score=0.3, weight=0.5),
    ]
    ok, failures = passes_floors({"mmlu": 0.6, "gsm8k": 0.4}, tasks)
    assert ok is True and failures == []

    ok, failures = passes_floors({"mmlu": 0.4, "gsm8k": 0.4}, tasks)
    assert ok is False and failures == ["mmlu"]

    # missing task counts as 0.0 -> fails its floor
    ok, failures = passes_floors({"mmlu": 0.6}, tasks)
    assert ok is False and failures == ["gsm8k"]


def test_passes_memory_cap_only_enforced_for_ram_ceiling():
    benchmark_floor = make_benchmark_floor_spec()
    assert passes_memory_cap(999999, benchmark_floor) is True  # not enforced

    ceiling = make_ram_ceiling_spec()
    assert passes_memory_cap(1000, ceiling) is True
    assert passes_memory_cap(1001, ceiling) is False


def make_submission(max_memory: int, scores: dict) -> MinerSubmission:
    return MinerSubmission(
        competition_id="rf",
        claims=[Claim(b=name, s=score) for name, score in scores.items()],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=max_memory,
        huggingface_revision="a" * 40,
    )


def test_sort_by_self_reported_benchmark_floor_lowest_memory_first():
    spec = make_benchmark_floor_spec()
    submissions = {
        "hk_big": make_submission(2000, {"mmlu": 0.9, "gsm8k": 0.9}),
        "hk_small": make_submission(1000, {"mmlu": 0.5, "gsm8k": 0.5}),
    }
    ranked = sort_by_self_reported(submissions, spec)
    assert [hk for hk, _ in ranked] == ["hk_small", "hk_big"]


def test_sort_by_self_reported_ram_ceiling_highest_composite_first():
    spec = make_ram_ceiling_spec()
    submissions = {
        "hk_low": make_submission(500, {"mmlu": 0.4, "gsm8k": 0.4}),
        "hk_high": make_submission(500, {"mmlu": 0.9, "gsm8k": 0.9}),
    }
    ranked = sort_by_self_reported(submissions, spec)
    assert [hk for hk, _ in ranked] == ["hk_high", "hk_low"]


def make_result(hotkey: str, final: float, passed=True, disqualified=False) -> ScoringResult:
    return ScoringResult(
        hotkey=hotkey,
        competition_id="rf",
        passed_floors=passed,
        disqualified=disqualified,
        actual_scores={},
        final_score=final,
        max_memory_kb=1000,
        lying_detected=False,
    )


def test_compute_emission_weights_ranks_and_zeroes_non_qualifiers():
    ranked = [
        make_result("hk1", 3.0),
        make_result("hk2", 2.0),
        make_result("hk3", 1.0, passed=False),
        make_result("hk4", 0.5, disqualified=True),
    ]
    weights = compute_emission_weights(ranked, [0.7, 0.3])
    assert weights == {"hk1": 0.7, "hk2": 0.3, "hk3": 0.0, "hk4": 0.0}


def test_compute_emission_weights_more_qualifiers_than_distribution_slots():
    ranked = [make_result("hk1", 3.0), make_result("hk2", 2.0)]
    weights = compute_emission_weights(ranked, [1.0])
    assert weights == {"hk1": 1.0, "hk2": 0.0}
