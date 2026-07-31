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


def test_phase_open_before_commit_end():
    spec = make_spec()
    assert spec.phase(99) == CompetitionPhase.OPEN


def test_phase_revealing_at_commit_end():
    spec = make_spec()
    assert spec.phase(100) == CompetitionPhase.REVEALING


def test_phase_revealing_until_grace_ends():
    spec = make_spec()
    assert spec.phase(109) == CompetitionPhase.REVEALING


def test_phase_scoring_after_grace():
    spec = make_spec()
    assert spec.phase(110) == CompetitionPhase.SCORING


def test_phase_distributing_after_scoring_end():
    spec = make_spec(distribution_blocks=5)
    assert spec.phase(200) == CompetitionPhase.DISTRIBUTING


def test_phase_complete_after_distribution_end():
    spec = make_spec(distribution_blocks=5)
    assert spec.phase(205) == CompetitionPhase.COMPLETE


def test_blocks_until_next_phase_during_revealing():
    spec = make_spec()
    assert spec.blocks_until_next_phase(105) == 5
