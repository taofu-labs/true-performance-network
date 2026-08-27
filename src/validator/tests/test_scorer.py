import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from common import settings as common_settings
from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import Claim, MinerSubmission
from competition.benchmark_client import MockCoordinator
from validator import scorer, store


def iso_ago(seconds: float) -> str:
    """ISO-8601 UTC timestamp `seconds` in the past, as the coordinator renders it."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def make_spec(**overrides) -> CompetitionSpec:
    fields = dict(
        id="comp1", name="comp1", start_block=0, commit_end_block=10, scoring_end_block=20,
        emission_distribution=[1.0], top_n=1,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5, weight=1.0)],
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


def make_submission(**overrides) -> MinerSubmission:
    fields = dict(
        competition_id="comp1",
        claims=[Claim(b="mmlu", s=0.7)],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        huggingface_revision="a" * 40,
    )
    fields.update(overrides)
    return MinerSubmission(**fields)


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(store._SCHEMA)
    return conn


def insert_candidate(conn, hotkey, submission, status="standby", **extra):
    store.insert_revealed_candidate(
        conn, "comp1", hotkey, rank=0, submission_json=submission.model_dump_json(),
        reveal_block=5, status=status,
    )
    conn.commit()
    if extra:
        conn.execute(
            "UPDATE revealed_candidates SET " + ", ".join(f"{k} = ?" for k in extra) +
            " WHERE competition_id = 'comp1' AND hotkey = ?",
            (*extra.values(), hotkey),
        )
        conn.commit()


class FakeContainer:
    """Precheck container stub used for tests that only need to get past the
    early guards, not exercise real RAM/sha256/provenance checks."""
    def check(self, url, context_length):
        from competition.precheck_client import PrecheckVerdict
        return PrecheckVerdict(provenance=None, ram=None, sha256="a" * 64)


# ---------------------------------------------------------------------------
# precheck_one
# ---------------------------------------------------------------------------

def test_precheck_one_fails_when_repo_not_public(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: False)
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), FakeContainer(), make_db())
    assert result.passed is False
    assert "not publicly accessible" in result.reason


def test_precheck_one_fails_when_no_gguf_file(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["config.json", "README.md"])
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), FakeContainer(), make_db())
    assert result.passed is False
    assert "no .gguf file" in result.reason


def test_precheck_one_passes_with_measured_ram(monkeypatch):
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class RamContainer:
        def check(self, url, context_length):
            return PrecheckVerdict(provenance=None, sha256="a" * 64, ram=RamResult(passed=True, ram_bytes=1000 * 1024))

    result = scorer.precheck_one("hk1", make_submission(), make_spec(), RamContainer(), make_db())
    assert result.passed is True
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
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), MismatchContainer(), conn)
    assert result.passed is False
    assert "hotkey banned" in result.reason
    assert store.is_banned(conn, "hk1") is True


def test_precheck_one_disqualifies_on_max_memory_lie(monkeypatch):
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class LyingRamContainer:
        def check(self, url, context_length):
            return PrecheckVerdict(provenance=None, sha256="a" * 64, ram=RamResult(passed=True, ram_bytes=5_000_000))

    submission = make_submission(max_memory=1000)  # 1000 KB reported, 5MB ~ 4883 KB measured
    result = scorer.precheck_one("hk1", submission, make_spec(), LyingRamContainer(), make_db())
    assert result.passed is False
    assert "max_memory lie" in result.reason


# ---------------------------------------------------------------------------
# dedup_winner
# ---------------------------------------------------------------------------

def test_dedup_winner_no_collision_when_hash_unseen():
    assert scorer.dedup_winner({}, "hk1", "a" * 64, 100) is None


def test_dedup_winner_later_reveal_block_loses():
    seen = {"a" * 64: ("hkA", 100)}
    assert scorer.dedup_winner(seen, "hkB", "a" * 64, 105) == ("hkA", 100)


def test_dedup_winner_earlier_reveal_block_wins():
    seen = {"a" * 64: ("hkA", 105)}
    assert scorer.dedup_winner(seen, "hkB", "a" * 64, 100) is None


def test_dedup_winner_tie_break_by_hotkey():
    seen = {"a" * 64: ("hkB", 100)}
    assert scorer.dedup_winner(seen, "hkA", "a" * 64, 100) is None
    assert scorer.dedup_winner(seen, "hkC", "a" * 64, 100) == ("hkB", 100)


# ---------------------------------------------------------------------------
# submit_benchmarks_for_candidate
# ---------------------------------------------------------------------------

def test_submit_benchmarks_persists_run_ids():
    conn = make_db()
    submission = make_submission()
    coordinator = MockCoordinator()

    err = scorer.submit_benchmarks_for_candidate(
        conn, "comp1", "hk1", submission, "model.gguf", make_spec(), coordinator,
    )
    assert err is None

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert len(rows) == 1
    assert rows[0]["benchmark_name"] == "mmlu"
    assert rows[0]["status"] == "submitted"
    assert rows[0]["coordinator_run_id"]


def test_submit_benchmarks_returns_error_on_submit_failure():
    conn = make_db()
    submission = make_submission()

    class FailingCoordinator:
        def submit(self, *a, **k):
            raise RuntimeError("coordinator unreachable")

    err = scorer.submit_benchmarks_for_candidate(
        conn, "comp1", "hk1", submission, "model.gguf", make_spec(), FailingCoordinator(),
    )
    assert err is not None
    assert "benchmark submit error" in err


# ---------------------------------------------------------------------------
# poll_open_benchmarks
# ---------------------------------------------------------------------------

def _mock_score(repo: str, revision: str, benchmark: str) -> float:
    """Mirror MockCoordinator's deterministic score derivation."""
    import hashlib
    import uuid
    seed = f"{repo}:{revision}:{benchmark}"
    run_id = str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))
    h = int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
    return round(0.55 + (h % 10000) / 10000 * 0.30, 4)


def test_poll_open_benchmarks_scores_and_marks_done():
    """The core C19 fix: score is persisted on benchmark_results the instant
    it's observed complete, not just the status, and the candidate resolves
    to 'done' with a scoring_results row."""
    conn = make_db()
    score = _mock_score("user/repo", "a" * 40, "mmlu")
    submission = make_submission(claims=[Claim(b="mmlu", s=score)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)

    coordinator = MockCoordinator()
    scorer.submit_benchmarks_for_candidate(conn, "comp1", "hk1", submission, "model.gguf", make_spec(), coordinator)

    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", make_spec(), coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"
    results = store.scoring_results_for_competition(conn, "comp1")
    assert len(results) == 1
    assert results[0]["hotkey"] == "hk1"

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] is not None


def test_poll_open_benchmarks_skips_completed_rows_no_resubmit():
    """A benchmark already 'completed' (score persisted) must never be
    touched again by poll_open_benchmarks — it's not in open_benchmark_results."""
    conn = make_db()
    submission = make_submission()
    insert_candidate(conn, "hk1", submission, status="benchmarking")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    conn.commit()
    store.update_benchmark_result(conn, "comp1", "hk1", "mmlu", "completed", score=0.9)
    conn.commit()

    class ExplodingCoordinator:
        def poll(self, run_id):
            raise AssertionError("must not poll an already-completed benchmark")

    scorer.poll_open_benchmarks(conn, "comp1", make_spec(), ExplodingCoordinator())
    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["score"] == 0.9


def test_poll_open_benchmarks_fails_candidate_below_floor():
    conn = make_db()
    spec = make_spec(benchmarks=[BenchmarkTask(name="mmlu", min_score=0.999, weight=1.0)])
    score = _mock_score("user/repo", "a" * 40, "mmlu")
    submission = make_submission(claims=[Claim(b="mmlu", s=score)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)

    coordinator = MockCoordinator()
    scorer.submit_benchmarks_for_candidate(conn, "comp1", "hk1", submission, "model.gguf", spec, coordinator)
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]


def test_poll_open_benchmarks_bans_when_actual_lower_than_claimed():
    conn = make_db()
    # Claim well above the mock's actual deterministic score.
    submission = make_submission(claims=[Claim(b="mmlu", s=0.99)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)

    coordinator = MockCoordinator()
    scorer.submit_benchmarks_for_candidate(conn, "comp1", "hk1", submission, "model.gguf", make_spec(), coordinator)
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", make_spec(), coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "hotkey banned" in candidate["failure_reason"]
    assert store.is_banned(conn, "hk1") is True


def test_poll_open_benchmarks_does_not_ban_when_actual_higher_than_claimed():
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.01)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)

    coordinator = MockCoordinator()
    scorer.submit_benchmarks_for_candidate(conn, "comp1", "hk1", submission, "model.gguf", make_spec(), coordinator)
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", make_spec(), coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"
    assert store.is_banned(conn, "hk1") is False


class NeverCompletingCoordinator:
    """Always RUNNING with a last_log_at that never advances — simulates a
    truly stalled coordinator run (no progress ever reported)."""
    def __init__(self, log_age_seconds: float = 3600):
        self._last_log_at = iso_ago(log_age_seconds)

    def submit(self, *a, **k):
        return "run-pending"

    def poll(self, run_id):
        from competition.benchmark_client import RunStatus, RunStatusCode
        return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="benchmarking",
                         last_log_at=self._last_log_at)


def test_poll_open_benchmarks_fails_when_log_older_than_window(monkeypatch):
    monkeypatch.setattr(common_settings, "BENCHMARK_EXECUTION_STALE_SECONDS", 60)
    conn = make_db()
    insert_candidate(conn, "hk1", make_submission(), status="benchmarking")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-pending")
    conn.commit()

    scorer.poll_open_benchmarks(conn, "comp1", make_spec(), NeverCompletingCoordinator(log_age_seconds=3600))

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "failed"
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "stalled" in candidate["failure_reason"]


def test_poll_open_benchmarks_keeps_running_when_log_within_window(monkeypatch):
    monkeypatch.setattr(common_settings, "BENCHMARK_EXECUTION_STALE_SECONDS", 3600)
    conn = make_db()
    insert_candidate(conn, "hk1", make_submission(), status="benchmarking")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-pending")
    conn.commit()

    scorer.poll_open_benchmarks(conn, "comp1", make_spec(), NeverCompletingCoordinator(log_age_seconds=5))

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "submitted"
    assert store.get_candidate(conn, "comp1", "hk1")["status"] == "benchmarking"


def test_poll_open_benchmarks_never_stalls_without_a_usable_timestamp(monkeypatch):
    """No last_log_at (or an unparseable one) means age is unmeasurable — a run
    must never be failed on a stall we cannot actually observe."""
    monkeypatch.setattr(common_settings, "BENCHMARK_EXECUTION_STALE_SECONDS", 0)

    class NoTimestamp:
        def __init__(self, value):
            self.value = value

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING,
                             phase="benchmarking", last_log_at=self.value)

    for value in (None, "", "not-a-timestamp"):
        conn = make_db()
        insert_candidate(conn, "hk1", make_submission(), status="benchmarking")
        store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-x")
        conn.commit()

        scorer.poll_open_benchmarks(conn, "comp1", make_spec(), NoTimestamp(value))

        rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
        assert rows[0]["status"] == "submitted", f"{value!r} should not stall"


def test_poll_open_benchmarks_future_timestamp_clamps_to_zero_age(monkeypatch):
    """A coordinator clock slightly ahead of ours must not read as a huge
    negative age, nor stall the run."""
    monkeypatch.setattr(common_settings, "BENCHMARK_EXECUTION_STALE_SECONDS", 0)
    conn = make_db()
    insert_candidate(conn, "hk1", make_submission(), status="benchmarking")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-future")
    conn.commit()

    scorer.poll_open_benchmarks(conn, "comp1", make_spec(), NeverCompletingCoordinator(log_age_seconds=-30))

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "submitted"


def test_poll_open_benchmarks_survives_long_execution_phase_with_fresh_progress():
    """A benchmark whose last_log_at keeps advancing must never be killed by
    elapsed time alone."""
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-slow")
    conn.commit()

    class SlowButProgressingCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 5:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="benchmarking", last_log_at=f"tick-{self._n}")
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = SlowButProgressingCoordinator()
    spec = make_spec()
    for _ in range(6):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"


def test_poll_open_benchmarks_estimated_seconds_remaining_extends_stale_window(monkeypatch):
    """A run reporting a large estimated_seconds_remaining must survive
    longer than the static per-bucket stale window before being called stale."""
    monkeypatch.setattr(common_settings, "BENCHMARK_EXECUTION_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-estimate")
    conn.commit()

    class LongEstimateCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 5:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="benchmarking", estimated_seconds_remaining=1_000_000)
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = LongEstimateCoordinator()
    spec = make_spec()
    for _ in range(6):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"


def test_poll_open_benchmarks_execution_bucket_survives_zero_startup_window(monkeypatch):
    """phase="benchmarking" must use BENCHMARK_EXECUTION_STALE_SECONDS, not
    BENCHMARK_STARTUP_STALE_SECONDS."""
    monkeypatch.setattr(common_settings, "BENCHMARK_STARTUP_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-execution")
    conn.commit()

    class ExecutionPhaseCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 3:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="benchmarking",
                                 last_log_at=iso_ago(30))
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = ExecutionPhaseCoordinator()
    spec = make_spec()
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"


def test_poll_open_benchmarks_queued_bucket_survives_zero_startup_window(monkeypatch):
    """phase="queued" must use BENCHMARK_QUEUED_STALE_SECONDS, not
    BENCHMARK_STARTUP_STALE_SECONDS — a run merely waiting for a worker slot
    must not be flagged stale just because the (unrelated) startup window is
    tight."""
    monkeypatch.setattr(common_settings, "BENCHMARK_STARTUP_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-queued")
    conn.commit()

    class QueuedPhaseCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 3:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="queued",
                                 last_log_at=iso_ago(30))
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = QueuedPhaseCoordinator()
    spec = make_spec()
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"


def test_poll_open_benchmarks_retry_waiting_bucket_survives_zero_startup_window(monkeypatch):
    """phase="retry_waiting" (status stays "queued" but phase is relabeled
    when a transiently-failed run is auto-requeued, see
    schedule_run_retry in the coordinator) must use
    BENCHMARK_QUEUED_STALE_SECONDS, not BENCHMARK_STARTUP_STALE_SECONDS —
    same frozen-last_log_at situation as plain "queued"."""
    monkeypatch.setattr(common_settings, "BENCHMARK_STARTUP_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-retry-waiting")
    conn.commit()

    class RetryWaitingPhaseCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 3:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="retry_waiting",
                                 last_log_at=iso_ago(30))
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = RetryWaitingPhaseCoordinator()
    spec = make_spec()
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"


def test_poll_open_benchmarks_queued_goes_stale_after_queued_window(monkeypatch):
    """A run whose phase stays "queued" forever (last_log_at frozen, as the
    coordinator never re-logs a merely-waiting run) must still eventually be
    caught once BENCHMARK_QUEUED_STALE_SECONDS itself elapses."""
    monkeypatch.setattr(common_settings, "BENCHMARK_QUEUED_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission()
    insert_candidate(conn, "hk1", submission, status="benchmarking")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-queued-stale")
    conn.commit()

    class NeverLeavingQueueCoordinator:
        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="queued",
                                 last_log_at=iso_ago(30))

    scorer.poll_open_benchmarks(conn, "comp1", make_spec(), NeverLeavingQueueCoordinator())

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "failed"
    assert store.get_candidate(conn, "comp1", "hk1")["status"] == "failed"


def test_poll_open_benchmarks_lm_eval_running_bucket_survives_zero_startup_window(monkeypatch):
    """phase="lm_eval_running" is the fine-grained sub-phase the coordinator
    reports while status="benchmarking" — it must use
    BENCHMARK_EXECUTION_STALE_SECONDS, not BENCHMARK_STARTUP_STALE_SECONDS.
    Previously this fell through to the startup bucket since only the
    coarse "benchmarking" string was recognized, not the fine-grained phase
    actually reported by /status."""
    monkeypatch.setattr(common_settings, "BENCHMARK_STARTUP_STALE_SECONDS", 0)
    conn = make_db()
    submission = make_submission(claims=[Claim(b="mmlu", s=0.7)])
    insert_candidate(conn, "hk1", submission, status="benchmarking", measured_memory_kb=999)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-lm-eval-running")
    conn.commit()

    class LmEvalRunningPhaseCoordinator:
        def __init__(self):
            self._n = 0

        def poll(self, run_id):
            from competition.benchmark_client import RunStatus, RunStatusCode
            self._n += 1
            if self._n < 3:
                return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, phase="lm_eval_running",
                                 last_log_at=iso_ago(30))
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.7})

    coordinator = LmEvalRunningPhaseCoordinator()
    spec = make_spec()
    for _ in range(4):
        scorer.poll_open_benchmarks(conn, "comp1", spec, coordinator)

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"
