import sqlite3

import pytest

from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import Claim, MinerSubmission
from competition.benchmark_client import MockCoordinator
from validator import scorer, store


def make_spec() -> CompetitionSpec:
    return CompetitionSpec(
        id="comp1",
        name="comp1",
        start_block=0,
        commit_end_block=10,
        scoring_end_block=20,
        emission_distribution=[1.0],
        top_n=1,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
    )


def make_submission() -> MinerSubmission:
    return MinerSubmission(
        competition_id="comp1",
        claims=[Claim(b="mmlu", s=0.7)],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        huggingface_revision="a" * 40,
    )


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(store._SCHEMA)
    return conn


class FakeContainer:
    """Precheck container stub used for tests that only need to get past the
    'no container' guard, not exercise real RAM/sha256/provenance checks."""
    def check(self, url, context_length):
        from competition.precheck_client import PrecheckVerdict
        return PrecheckVerdict(provenance=None, ram=None, sha256="a" * 64)


def test_precheck_one_skips_when_container_unavailable(monkeypatch):
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), precheck_ctr=None, conn=make_db())
    assert isinstance(result, scorer.ScoringOutcome)
    assert result.kind == scorer.OutcomeKind.SKIPPED
    assert "precheck container unavailable" in result.reason


def test_precheck_one_skips_when_repo_not_public(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: False)
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), precheck_ctr=FakeContainer(), conn=make_db())
    assert isinstance(result, scorer.ScoringOutcome)
    assert result.kind == scorer.OutcomeKind.SKIPPED
    assert "not publicly accessible" in result.reason


def test_precheck_one_skips_when_no_gguf_file(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["config.json", "README.md"])
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), precheck_ctr=FakeContainer(), conn=make_db())
    assert result.kind == scorer.OutcomeKind.SKIPPED
    assert "no .gguf file" in result.reason


def test_precheck_one_passes_with_measured_ram(monkeypatch):
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class RamContainer:
        def check(self, url, context_length):
            return PrecheckVerdict(
                provenance=None, sha256="a" * 64,
                ram=RamResult(passed=True, ram_bytes=1000 * 1024),
            )

    result = scorer.precheck_one("hk1", make_submission(), make_spec(), precheck_ctr=RamContainer(), conn=make_db())
    assert isinstance(result, scorer.PrecheckPass)
    assert result.gguf_file == "model.gguf"
    assert result.measured_memory_kb == 1000


def test_precheck_one_bans_on_sha256_mismatch(monkeypatch):
    from competition.precheck_client import PrecheckVerdict

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class MismatchContainer:
        def check(self, url, context_length):
            return PrecheckVerdict(provenance=None, ram=None, sha256="f" * 64)

    conn = make_db()
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), precheck_ctr=MismatchContainer(), conn=conn)
    assert result.kind == scorer.OutcomeKind.SKIPPED
    assert "hotkey banned" in result.reason
    assert store.is_banned(conn, "hk1") is True


def test_dedup_winner_no_collision_when_hash_unseen():
    result = scorer.dedup_winner({}, "hk1", "a" * 64, 100)
    assert result is None


def test_dedup_winner_later_reveal_block_loses():
    seen = {"a" * 64: ("hkA", 100)}
    result = scorer.dedup_winner(seen, "hkB", "a" * 64, 105)
    assert result == ("hkA", 100)


def test_dedup_winner_earlier_reveal_block_wins():
    seen = {"a" * 64: ("hkA", 105)}
    result = scorer.dedup_winner(seen, "hkB", "a" * 64, 100)
    assert result is None


def test_dedup_winner_tie_break_by_hotkey():
    seen = {"a" * 64: ("hkB", 100)}
    # "hkA" < "hkB" lexicographically — challenger wins the tie
    assert scorer.dedup_winner(seen, "hkA", "a" * 64, 100) is None
    # "hkC" > "hkB" lexicographically — challenger loses the tie
    assert scorer.dedup_winner(seen, "hkC", "a" * 64, 100) == ("hkB", 100)


def _mock_score(repo: str, revision: str, benchmark: str) -> float:
    """Mirror MockCoordinator's deterministic score derivation so tests can
    submit a matching claim and avoid tripping the score-mismatch ban."""
    import hashlib
    import uuid
    seed = f"{repo}:{revision}:{benchmark}"
    run_id = str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))
    h = int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
    return round(0.55 + (h % 10000) / 10000 * 0.30, 4)


def make_submission_matching_mock(**overrides) -> MinerSubmission:
    """Like make_submission(), but claims the score MockCoordinator will
    actually produce, so tests that exercise benchmark_one's floor/score
    checks don't spuriously trip the score-mismatch ban."""
    score = _mock_score("user/repo", "a" * 40, "mmlu")
    kwargs = dict(
        competition_id="comp1",
        claims=[Claim(b="mmlu", s=score)],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        huggingface_revision="a" * 40,
    )
    kwargs.update(overrides)
    return MinerSubmission(**kwargs)


@pytest.mark.asyncio
async def test_benchmark_one_scores_with_mock_coordinator():
    coordinator = MockCoordinator()
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    outcome = await scorer.benchmark_one(p, make_spec(), coordinator, conn=make_db(), poll_interval=0)
    assert outcome.kind == scorer.OutcomeKind.SCORED
    assert outcome.result.hotkey == "hk1"
    assert "mmlu" in outcome.result.actual_scores
    assert outcome.result.max_memory_kb == 999  # measured value, not submission's self-reported 1000


@pytest.mark.asyncio
async def test_benchmark_one_skips_below_floor():
    spec = CompetitionSpec(
        id="comp1", name="comp1", start_block=0, commit_end_block=10, scoring_end_block=20,
        emission_distribution=[1.0], top_n=1,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.999, weight=1.0)],
    )
    coordinator = MockCoordinator()
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    outcome = await scorer.benchmark_one(p, spec, coordinator, conn=make_db(), poll_interval=0)
    assert outcome.kind == scorer.OutcomeKind.SKIPPED
    assert "failed floors" in outcome.reason


@pytest.mark.asyncio
async def test_benchmark_one_bans_when_actual_lower_than_claimed():
    coordinator = MockCoordinator()
    # Claim well above the mock's actual deterministic score (~0.7388).
    submission = make_submission_matching_mock(claims=[Claim(b="mmlu", s=0.99)])
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()
    outcome = await scorer.benchmark_one(p, make_spec(), coordinator, conn=conn, poll_interval=0)
    assert outcome.kind == scorer.OutcomeKind.SKIPPED
    assert "hotkey banned" in outcome.reason
    assert store.is_banned(conn, "hk1") is True


@pytest.mark.asyncio
async def test_benchmark_one_does_not_ban_when_actual_higher_than_claimed():
    coordinator = MockCoordinator()
    # Claim well below the mock's actual deterministic score (~0.7388) —
    # over-delivery is not a lie and must not be banned.
    submission = make_submission_matching_mock(claims=[Claim(b="mmlu", s=0.01)])
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()
    outcome = await scorer.benchmark_one(p, make_spec(), coordinator, conn=conn, poll_interval=0)
    assert outcome.kind == scorer.OutcomeKind.SCORED
    assert store.is_banned(conn, "hk1") is False
