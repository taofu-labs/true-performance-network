"""
CompetitionSpec's own validators.

These run on every path that builds a spec — the admin route, the bearer-auth
upsert, the miner CLI parsing the leader's JSON, and the leader re-reading its
own stored rows on each tick. A rule enforced only in the API's creation_errors
covers just the first of those.
"""
import pytest
from pydantic import ValidationError

from common.models.competition import (
    KL_RUN_NAME,
    BenchmarkTask,
    CompetitionSpec,
    CompetitionType,
)


def make_spec(**overrides) -> CompetitionSpec:
    fields = dict(
        id="c1", name="c1", start_block=0, commit_end_block=10, scoring_end_block=20,
        top_n=1, emission_distribution=[1.0],
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5)],
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


# ── KL base pin ──────────────────────────────────────────────────────────────

def test_kl_corpus_requires_a_base_repo():
    """Without model_repo the miner's comparison run names its own base, so they
    can pick a weak model their quant barely diverges from and the multiplier
    becomes free."""
    with pytest.raises(ValidationError, match="model_repo"):
        make_spec(
            kl_corpus_id="corpus-000002",
            competition_type=CompetitionType.GPU_MEMORY_CEILING,
            max_gpu_memory_kb=1000,
        )


def test_kl_corpus_with_a_base_repo_is_accepted():
    spec = make_spec(
        kl_corpus_id="corpus-000002",
        model_repo="Qwen/Qwen2.5-0.5B-Instruct",
        competition_type=CompetitionType.GPU_MEMORY_CEILING,
        max_gpu_memory_kb=1000,
    )
    assert spec.kl_corpus_id == "corpus-000002"
    assert spec.model_repo == "Qwen/Qwen2.5-0.5B-Instruct"


def test_a_base_repo_without_a_kl_corpus_is_fine():
    """model_repo documents what miners should optimize. It is only *required*
    when a KL run has to be verified against it."""
    assert make_spec(model_repo="Qwen/Qwen2.5-0.5B-Instruct").kl_corpus_id is None


def test_a_competition_with_neither_is_unaffected():
    spec = make_spec()
    assert spec.kl_corpus_id is None and spec.model_repo is None


# ── Sibling validators, previously untested ──────────────────────────────────

def test_kl_is_reserved_and_cannot_name_a_benchmark():
    """A benchmark named `kl` would collide with the multiplier in verified_scores."""
    with pytest.raises(ValidationError, match=KL_RUN_NAME):
        make_spec(benchmarks=[BenchmarkTask(name=KL_RUN_NAME, min_score=0.5)])


def test_gpu_memory_ceiling_requires_a_cap():
    with pytest.raises(ValidationError, match="max_gpu_memory_kb"):
        make_spec(competition_type=CompetitionType.GPU_MEMORY_CEILING)


def test_benchmark_floor_needs_no_cap():
    assert make_spec(competition_type=CompetitionType.BENCHMARK_FLOOR).max_gpu_memory_kb is None


def test_emission_distribution_must_match_top_n_and_sum_to_one():
    with pytest.raises(ValidationError, match="top_n"):
        make_spec(top_n=2, emission_distribution=[1.0])
    with pytest.raises(ValidationError, match="sums to"):
        make_spec(top_n=2, emission_distribution=[0.5, 0.2])
