import asyncio
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
async def test_benchmark_one_persists_run_id_after_submit():
    """C11-C14 finding #2: coordinator run_id must be persisted immediately
    after submit, not only held in-memory, so a killed/restarted validator
    can resume polling instead of resubmitting."""
    coordinator = MockCoordinator()
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()
    await scorer.benchmark_one(p, make_spec(), coordinator, conn=conn, poll_interval=0)

    pending = store.pending_benchmark_runs(conn, "comp1")
    # completed by the time benchmark_one returns, so no longer "pending" —
    # but the row must exist with a terminal status, proving it was persisted.
    row = conn.execute(
        "SELECT * FROM benchmark_runs WHERE competition_id = ? AND hotkey = ?", ("comp1", "hk1")
    ).fetchone()
    assert row is not None
    assert row["status"] == "completed"
    assert row["coordinator_run_id"]


@pytest.mark.asyncio
async def test_benchmark_one_skips_submit_for_resumed_run_ids():
    """A run_id passed in via `run_ids` (resumed from a prior process) must
    not be resubmitted — only polled."""
    coordinator = MockCoordinator()
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()

    # Pre-submit once to get a real run_id from the mock, simulating a prior process.
    existing_run_id = coordinator.submit("user/repo", "a" * 40, "mmlu")
    submit_calls = {"n": 0}
    orig_submit = coordinator.submit

    def counting_submit(*a, **k):
        submit_calls["n"] += 1
        return orig_submit(*a, **k)
    coordinator.submit = counting_submit

    outcome = await scorer.benchmark_one(
        p, make_spec(), coordinator, conn=conn, poll_interval=0,
        run_ids={"mmlu": existing_run_id},
    )
    assert submit_calls["n"] == 0  # resumed, not resubmitted
    assert outcome.kind == scorer.OutcomeKind.SCORED


@pytest.mark.asyncio
async def test_benchmark_one_marks_pending_resume_when_window_still_open():
    """C11-C14 finding #2: a poll timeout while the scoring window is still
    open must not be a hard skip — it should be left resumable for the next
    leader-loop tick, not converted to a score-0 participant failure."""
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()

    class NeverCompletingCoordinator:
        def submit(self, *a, **k):
            return "run-pending"

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING)

    spec = make_spec()  # scoring_end_block=20
    outcome = await scorer.benchmark_one(
        p, spec, NeverCompletingCoordinator(), conn=conn, poll_interval=0,
        max_wait_seconds=0, current_block=15,  # 15 < scoring_end_block=20 -> window open
    )
    assert outcome.kind == scorer.OutcomeKind.SKIPPED
    assert "pending-resume" in outcome.reason

    row = conn.execute(
        "SELECT status FROM benchmark_runs WHERE competition_id = ? AND hotkey = ?", ("comp1", "hk1")
    ).fetchone()
    assert row["status"] == "pending-resume"


@pytest.mark.asyncio
async def test_benchmark_one_hard_skips_when_window_closed():
    """Once the scoring window has actually closed, a still-pending poll
    must finalize as a real timeout/skip — no more resuming."""
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()

    class NeverCompletingCoordinator:
        def submit(self, *a, **k):
            return "run-pending"

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING)

    spec = make_spec()  # scoring_end_block=20
    outcome = await scorer.benchmark_one(
        p, spec, NeverCompletingCoordinator(), conn=conn, poll_interval=0,
        max_wait_seconds=0, current_block=25,  # 25 >= scoring_end_block=20 -> window closed
    )
    assert outcome.kind == scorer.OutcomeKind.SKIPPED
    assert "pending-resume" not in outcome.reason
    assert "poll timeout" in outcome.reason


@pytest.mark.asyncio
async def test_benchmark_one_respects_submit_semaphore():
    """C11-C14 finding #4: submit concurrency must be boundable independent
    of poll concurrency."""
    submission = make_submission_matching_mock()
    p = scorer.PrecheckPass(hotkey="hk1", submission=submission, gguf_file="model.gguf", measured_memory_kb=999)
    conn = make_db()

    max_concurrent = {"seen": 0, "current": 0}

    class TrackingCoordinator(MockCoordinator):
        def submit(self, *a, **k):
            max_concurrent["current"] += 1
            max_concurrent["seen"] = max(max_concurrent["seen"], max_concurrent["current"])
            try:
                return super().submit(*a, **k)
            finally:
                max_concurrent["current"] -= 1

    sem = asyncio.Semaphore(1)
    coordinator = TrackingCoordinator()
    outcomes = await asyncio.gather(*[
        scorer.benchmark_one(
            scorer.PrecheckPass(hotkey=f"hk{i}", submission=submission, gguf_file="model.gguf", measured_memory_kb=999),
            make_spec(), coordinator, conn=conn, poll_interval=0, submit_semaphore=sem,
        )
        for i in range(3)
    ])
    assert max_concurrent["seen"] == 1
    assert all(o.kind == scorer.OutcomeKind.SCORED for o in outcomes)


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
