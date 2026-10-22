import pytest

from common.models.competition import KL_RUN_NAME, BenchmarkTask, CompetitionSpec, CompetitionType
from common.models.submission import BenchmarkRun, MinerSubmission, ScoringResult
from competition.scoring import (
    aggregate_competition_weights,
    benchmark_min,
    compute_emission_weights,
    final_score,
    kl_norm,
    passes_floors,
    passes_memory_cap,
    sort_by_verified_scores,
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
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5)],
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
            BenchmarkTask(name="mmlu", min_score=0.5),
            BenchmarkTask(name="gsm8k", min_score=0.3),
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
        competition_type=CompetitionType.GPU_MEMORY_CEILING,
        max_gpu_memory_kb=1000,
        benchmarks=[
            BenchmarkTask(name="mmlu", min_score=0.5),
            BenchmarkTask(name="gsm8k", min_score=0.3),
        ],
    )
    defaults.update(overrides)
    return CompetitionSpec(**defaults)


def test_benchmark_min_takes_the_weakest_benchmark():
    tasks = [
        BenchmarkTask(name="mmlu", min_score=0.0),
        BenchmarkTask(name="gsm8k", min_score=0.0),
    ]
    # A strong mmlu cannot carry a weak gsm8k — the min is what ranks.
    assert benchmark_min({"mmlu": 0.9, "gsm8k": 0.2}, tasks) == pytest.approx(0.2)


def test_benchmark_min_missing_task_counts_zero():
    tasks = [
        BenchmarkTask(name="mmlu", min_score=0.0),
        BenchmarkTask(name="gsm8k", min_score=0.0),
    ]
    assert benchmark_min({"mmlu": 0.8}, tasks) == 0.0  # gsm8k missing -> 0


def test_benchmark_min_no_tasks_returns_zero():
    assert benchmark_min({"mmlu": 0.9}, []) == 0.0


def test_final_score_benchmark_floor_is_negated_memory():
    spec = make_benchmark_floor_spec()
    assert final_score({}, 5000, spec) == -5000.0


def test_final_score_ram_ceiling_is_min_benchmark():
    spec = make_ram_ceiling_spec()
    assert final_score({"mmlu": 0.8, "gsm8k": 0.4}, 500, spec) == pytest.approx(0.4)


def test_passes_floors_all_pass_and_one_fail():
    tasks = [
        BenchmarkTask(name="mmlu", min_score=0.5),
        BenchmarkTask(name="gsm8k", min_score=0.3),
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


def make_submission() -> MinerSubmission:
    return MinerSubmission(
        competition_id="rf",
        runs=[BenchmarkRun(b="mmlu", r="r1"), BenchmarkRun(b="gsm8k", r="r2")],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        huggingface_revision="a" * 40,
    )


def test_sort_by_verified_scores_benchmark_floor_lowest_memory_first():
    """benchmark_floor ranks on the memory the coordinator measured, which the
    caller passes in alongside the verified scores — a miner cannot win this
    ranking by claiming a small number."""
    spec = make_benchmark_floor_spec()
    submissions = {"hk_big": make_submission(), "hk_small": make_submission()}
    verified = {
        "hk_big": {"mmlu": 0.9, "gsm8k": 0.9},
        "hk_small": {"mmlu": 0.5, "gsm8k": 0.5},
    }
    measured = {"hk_big": 2000, "hk_small": 1000}
    ranked = sort_by_verified_scores(submissions, verified, spec, measured)
    assert [hk for hk, _ in ranked] == ["hk_small", "hk_big"]


def test_sort_by_verified_scores_without_measured_memory_ranks_as_zero():
    """A hotkey with no measured figure ranks as 0 KB, which for benchmark_floor
    is the best possible slot — so the caller must only rank candidates whose
    memory is known. Pinned here so the hazard is not discovered in production."""
    spec = make_benchmark_floor_spec()
    submissions = {"hk_known": make_submission(), "hk_unknown": make_submission()}
    verified = {"hk_known": {"mmlu": 0.9, "gsm8k": 0.9}, "hk_unknown": {"mmlu": 0.9, "gsm8k": 0.9}}
    ranked = sort_by_verified_scores(submissions, verified, spec, {"hk_known": 1000})
    assert [hk for hk, _ in ranked] == ["hk_unknown", "hk_known"]


def test_sort_by_verified_scores_ram_ceiling_highest_min_first():
    spec = make_ram_ceiling_spec()
    submissions = {"hk_low": make_submission(), "hk_high": make_submission()}
    verified = {
        "hk_low": {"mmlu": 0.4, "gsm8k": 0.4},
        "hk_high": {"mmlu": 0.9, "gsm8k": 0.9},
    }
    ranked = sort_by_verified_scores(submissions, verified, spec)
    assert [hk for hk, _ in ranked] == ["hk_high", "hk_low"]



def test_sort_by_verified_scores_candidate_with_no_verified_scores_ranks_last():
    """Every run rejected -> all zeros -> bottom of the ranking, but still
    present so the floors can reject it with a record."""
    spec = make_ram_ceiling_spec()
    submissions = {"hk_ok": make_submission(), "hk_rejected": make_submission()}
    verified = {"hk_ok": {"mmlu": 0.7, "gsm8k": 0.7}, "hk_rejected": {"mmlu": 0.0, "gsm8k": 0.0}}
    ranked = sort_by_verified_scores(submissions, verified, spec)
    assert [hk for hk, _ in ranked] == ["hk_ok", "hk_rejected"]



def make_result(hotkey: str, final: float) -> ScoringResult:
    return ScoringResult(
        hotkey=hotkey,
        competition_id="rf",
        final_score=final,
        measured_memory_kb=1000,
    )


def test_compute_emission_weights_ranks_qualifiers():
    ranked = [make_result("hk1", 3.0), make_result("hk2", 2.0)]
    weights = compute_emission_weights(ranked, [0.7, 0.3])
    assert weights == {"hk1": 0.7, "hk2": 0.3}


def test_compute_emission_weights_more_qualifiers_than_distribution_slots():
    ranked = [make_result("hk1", 3.0), make_result("hk2", 2.0)]
    weights = compute_emission_weights(ranked, [1.0])
    assert weights == {"hk1": 1.0, "hk2": 0.0}


# ── KL score ──────────────────────────────────────────────────────────────────
# KL anchors are measured P99 on control_to_candidate (nats): Q4_K_M 0.1455,
# Q2_K 1.30.

def make_kl_spec(competition_type=CompetitionType.GPU_MEMORY_CEILING) -> CompetitionSpec:
    return CompetitionSpec(
        id="kl", name="kl", kl_corpus_id="corpus-000002",
        # A KL corpus needs a base to compare against; the spec refuses one without.
        model_repo="Qwen/Qwen2.5-0.5B-Instruct",
        start_block=0, commit_end_block=10, scoring_end_block=20,
        top_n=1, emission_distribution=[1.0],
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.0), BenchmarkTask(name="gsm8k", min_score=0.0)],
        competition_type=competition_type,
        max_gpu_memory_kb=1000 if competition_type == CompetitionType.GPU_MEMORY_CEILING else None,
    )


def test_kl_norm_is_one_at_zero_divergence():
    assert kl_norm(0.0) == 1.0


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_kl_norm_rejects_impossible_values(bad):
    assert kl_norm(bad) == 0.0


def test_kl_norm_is_monotonic_over_the_measured_ladder():
    ladder = [0.0, 0.011, 0.14554, 0.59, 1.30]
    assert all(kl_norm(a) > kl_norm(b) for a, b in zip(ladder, ladder[1:]))


def test_kl_norm_barely_penalizes_q8_and_heavily_penalizes_q2k():
    assert kl_norm(0.011) > 0.98
    assert kl_norm(0.14554) == pytest.approx(0.8646, abs=1e-4)
    assert kl_norm(1.30) < 0.30


def test_kl_multiplier_discounts_the_weakest_benchmark():
    spec = make_kl_spec()
    scores = {"mmlu": 0.9, "gsm8k": 0.6, KL_RUN_NAME: 0.5}
    assert final_score(scores, 500, spec) == pytest.approx(0.30)


def test_kl_key_is_never_read_as_a_benchmark():
    spec = make_kl_spec()
    scores = {"mmlu": 0.9, "gsm8k": 0.6, KL_RUN_NAME: 0.01}
    assert benchmark_min(scores, spec.benchmarks) == 0.6
    assert passes_floors(scores, spec.benchmarks) == (True, [])


def test_absent_kl_score_leaves_ranking_unchanged():
    spec = make_kl_spec()
    assert final_score({"mmlu": 0.9, "gsm8k": 0.6}, 500, spec) == 0.6


def test_unverifiable_kl_run_zeroes_the_candidate():
    spec = make_kl_spec()
    assert final_score({"mmlu": 0.9, "gsm8k": 0.6, KL_RUN_NAME: 0.0}, 500, spec) == 0.0


def test_benchmark_floor_ignores_the_kl_multiplier():
    spec = make_kl_spec(CompetitionType.BENCHMARK_FLOOR)
    scores = {"mmlu": 0.9, "gsm8k": 0.6, KL_RUN_NAME: 0.1}
    assert final_score(scores, 500, spec) == -500.0


def test_kl_reorders_ranking_against_a_damaged_model():
    spec = make_kl_spec()
    submissions = {
        "honest": MinerSubmission(
            competition_id="kl", runs=[BenchmarkRun(b="mmlu", r="r1")], repository="a/b",
            file="f.gguf", file_sha256="a" * 64, max_memory=500, huggingface_revision="b" * 40,
        ),
        "damaged": MinerSubmission(
            competition_id="kl", runs=[BenchmarkRun(b="mmlu", r="r2")], repository="c/d",
            file="g.gguf", file_sha256="c" * 64, max_memory=500, huggingface_revision="d" * 40,
        ),
    }
    verified = {
        "honest": {"mmlu": 0.70, "gsm8k": 0.70, KL_RUN_NAME: kl_norm(0.011)},
        "damaged": {"mmlu": 0.85, "gsm8k": 0.85, KL_RUN_NAME: kl_norm(1.30)},
    }
    assert [hk for hk, _ in sort_by_verified_scores(submissions, verified, spec)] == ["honest", "damaged"]


def test_kl_is_reserved_and_cannot_name_a_benchmark():
    with pytest.raises(ValueError, match="reserved"):
        CompetitionSpec(
            id="x", name="x", start_block=0, commit_end_block=10, scoring_end_block=20,
            top_n=1, emission_distribution=[1.0],
            benchmarks=[BenchmarkTask(name=KL_RUN_NAME, min_score=0.5)],
        )
