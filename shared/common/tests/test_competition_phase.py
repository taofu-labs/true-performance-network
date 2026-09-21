import pytest

from common.models.competition import BenchmarkTask, CompetitionPhase, CompetitionSpec


def make_spec(**overrides) -> CompetitionSpec:
    fields = {
        "id": "comp1",
        "name": "comp1",
        "start_block": 0,
        "commit_end_block": 100,
        "scoring_end_block": 200,
        "reveal_grace_blocks": 10,
        "emission_distribution": [1.0],
        "top_n": 1,
        "benchmarks": [BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
    }
    fields.update(overrides)
    return CompetitionSpec(**fields)


@pytest.mark.parametrize("block,expected,overrides", [
    (99,  CompetitionPhase.OPEN,         {}),                        # before commit_end_block
    (100, CompetitionPhase.REVEALING,    {}),                        # commit_end_block itself
    (109, CompetitionPhase.REVEALING,    {}),                        # last block of the grace window
    (110, CompetitionPhase.SCORING,      {}),                        # grace elapsed
    (200, CompetitionPhase.DISTRIBUTING, {"distribution_blocks": 5}),  # scoring_end_block
    (205, CompetitionPhase.COMPLETE,     {"distribution_blocks": 5}),  # distribution window closed
    (200, CompetitionPhase.COMPLETE,     {}),                        # no distribution window at all
])
def test_phase_boundaries(block, expected, overrides):
    assert make_spec(**overrides).phase(block) == expected


def test_blocks_until_next_phase_during_revealing():
    spec = make_spec()
    assert spec.blocks_until_next_phase(105) == 5  # 110 - 105
