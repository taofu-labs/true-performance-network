import sqlite3

from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import BenchmarkRun, MinerSubmission
from competition.benchmark_client import RunStatus, RunStatusCode
from validator import scorer, store


def make_spec(**overrides) -> CompetitionSpec:
    fields = dict(
        id="comp1", name="comp1", start_block=0, commit_end_block=10, scoring_end_block=20,
        emission_distribution=[1.0], top_n=1,
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5)],
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


def make_submission(**overrides) -> MinerSubmission:
    fields = dict(
        competition_id="comp1",
        runs=[BenchmarkRun(b="mmlu", r="r100")],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        huggingface_revision="b" * 40,
    )
    fields.update(overrides)
    return MinerSubmission(**fields)


def make_status(**overrides) -> RunStatus:
    """A run that verifies cleanly against make_submission()."""
    fields = dict(
        run_id="r100",
        status=RunStatusCode.COMPLETED,
        scores={"mmlu": 0.8},
        repo="user/repo",
        revision="b" * 40,
        model_files=["model.gguf"],
        file_hashes={"model.gguf": ("sha256", "a" * 64)},
        benchmarks=["mmlu"],
        item_status={"mmlu": "completed"},
    )
    fields.update(overrides)
    return RunStatus(**fields)


def make_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(store._SCHEMA)
    return conn


class StubCoordinator:
    """Returns a canned status per run id; records what was polled."""

    def __init__(self, by_run_id: dict):
        self._by_run_id = by_run_id
        self.polled = []

    def poll(self, run_id):
        self.polled.append(run_id)
        status = self._by_run_id.get(run_id)
        if status is None:
            raise RuntimeError(f"unknown run {run_id}")
        return status


class FakeContainer:
    """Precheck container stub for tests that only need to pass the early guards."""
    def check(self, repository, revision, filename, context_length):
        from competition.precheck_client import PrecheckVerdict
        return PrecheckVerdict(provenance=None, ram=None, sha256="a" * 64)


# ---------------------------------------------------------------------------
# verify_run — binding a miner-supplied run to the committed model
#
# This is the security core of the rework: without it a run id proves nothing,
# and a miner can benchmark a strong model while committing a weak one.
# ---------------------------------------------------------------------------

def test_verify_run_accepts_matching_run():
    result = scorer.verify_run(make_submission(), "mmlu", "r100", make_status())
    assert result.ok is True
    assert result.score == 0.8
    assert result.reason == ""


def test_verify_run_rejects_repo_mismatch():
    """The headline attack: benchmark one repo, commit another."""
    status = make_status(repo="attacker/strong-model")
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert result.score == 0.0
    assert "repo mismatch" in result.reason
    # The repo the run actually used is recorded, so the rejection is legible
    # without re-querying the coordinator.
    assert result.repo == "attacker/strong-model"


def test_verify_run_repo_comparison_is_case_insensitive():
    """HF repo ids are case-insensitive; a case difference is not an attack."""
    status = make_status(repo="User/Repo")
    assert scorer.verify_run(make_submission(), "mmlu", "r100", status).ok is True


def test_verify_run_rejects_revision_mismatch():
    """Same repo, different commit — the miner benchmarked another version."""
    status = make_status(revision="c" * 40)
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "revision mismatch" in result.reason


def test_verify_run_rejects_when_committed_file_not_in_run():
    """Same repo and revision, but the run loaded a different .gguf — the
    second form of the substitution attack, within one repo."""
    status = make_status(model_files=["other-model.gguf"], file_hashes={})
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "not among run's model files" in result.reason
    assert "other-model.gguf" in result.reason


def test_verify_run_rejects_sha256_mismatch():
    status = make_status(file_hashes={"model.gguf": ("sha256", "f" * 64)})
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "file hash mismatch" in result.reason


def test_verify_run_skips_hash_check_for_non_sha256_algorithms():
    """HF serves a true content sha256 only for LFS files; `xet` and
    `git_blob` values are not comparable to the miner's file_sha256. Repo +
    revision + path already pin the content and precheck rehashes the real
    file, so a non-sha256 algorithm must skip the check, not fail the run."""
    for algorithm in ("xet", "git_blob"):
        status = make_status(file_hashes={"model.gguf": (algorithm, "not-a-sha256")})
        result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
        assert result.ok is True, f"{algorithm} should skip the hash check"



def test_verify_run_rejects_when_benchmark_not_covered_by_run():
    """A valid run for the wrong benchmark cannot be reused for this one."""
    status = make_status(benchmarks=["gsm8k"], item_status={"gsm8k": "completed"})
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "does not cover benchmark" in result.reason


def test_verify_run_rejects_failed_benchmark_item_in_suite():
    """A partially completed suite is terminal overall, but a child that
    failed must not be scored from whatever the parent reported."""
    status = make_status(
        benchmarks=["mmlu", "gsm8k"],
        item_status={"mmlu": "failed", "gsm8k": "completed"},
    )
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "did not complete in run" in result.reason


def test_verify_run_accepts_pinned_fluid_epoch():
    status = make_status(fluid_epoch_id="epoch-000001-r7")
    assert scorer.verify_run(make_submission(), "mmlu", "r100", status, "epoch-000001-r7").ok is True


def test_verify_run_rejects_other_or_missing_fluid_epoch():
    for epoch in ["epoch-000001-r5", None]:
        status = make_status(fluid_epoch_id=epoch)
        result = scorer.verify_run(make_submission(), "mmlu", "r100", status, "epoch-000001-r7")
        assert result.ok is False and result.score == 0.0, epoch
        assert "competition requires epoch-000001-r7" in result.reason


def test_verify_candidate_runs_applies_spec_fluid_epoch():
    spec = make_spec(benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5, fluid_epoch_id="epoch-000001-r7")])
    old = StubCoordinator({"r100": make_status(fluid_epoch_id="epoch-000001-r5")})
    [result] = scorer.verify_candidate_runs(make_submission(), spec, old)
    assert result.ok is False


def test_verify_run_rejects_failed_run():
    status = make_status(status=RunStatusCode.FAILED, failure_reason="out of memory")
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "out of memory" in result.reason


def test_verify_run_rejects_still_running_run():
    """Hard cutoff — scoring has started, so a run still in flight is out of
    time. Miners have the whole commit window to finish."""
    status = make_status(status=RunStatusCode.RUNNING, scores={})
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "not complete at scoring time" in result.reason


def test_verify_run_rejects_run_without_model_identity():
    """A coordinator response with no model_source cannot be bound to
    anything, so it must not be trusted by default."""
    status = make_status(repo=None)
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "no model identity" in result.reason


def test_verify_run_rejects_completed_run_missing_the_score():
    status = make_status(scores={})
    result = scorer.verify_run(make_submission(), "mmlu", "r100", status)
    assert result.ok is False
    assert "no score" in result.reason


# ---------------------------------------------------------------------------
# verify_candidate_runs
# ---------------------------------------------------------------------------

def test_verify_candidate_runs_covers_every_spec_benchmark():
    spec = make_spec(benchmarks=[
        BenchmarkTask(name="mmlu", min_score=0.5),
        BenchmarkTask(name="gsm8k", min_score=0.5),
    ])
    submission = make_submission(runs=[
        BenchmarkRun(b="mmlu", r="r100"),
        BenchmarkRun(b="gsm8k", r="r200"),
    ])
    coordinator = StubCoordinator({
        "r100": make_status(),
        "r200": make_status(run_id="r200", scores={"gsm8k": 0.6},
                            benchmarks=["gsm8k"], item_status={"gsm8k": "completed"}),
    })

    results = scorer.verify_candidate_runs(submission, spec, coordinator)
    assert {r.benchmark for r in results} == {"mmlu", "gsm8k"}
    assert all(r.ok for r in results)


def test_verify_candidate_runs_scores_zero_for_missing_run_id():
    """A benchmark the miner submitted no run for scores 0.0 — floors decide
    whether that is survivable, per decision 6."""
    spec = make_spec(benchmarks=[
        BenchmarkTask(name="mmlu", min_score=0.5),
        BenchmarkTask(name="gsm8k", min_score=0.5),
    ])
    submission = make_submission(runs=[BenchmarkRun(b="mmlu", r="r100")])
    coordinator = StubCoordinator({"r100": make_status()})

    results = {r.benchmark: r for r in scorer.verify_candidate_runs(submission, spec, coordinator)}
    assert results["gsm8k"].ok is False
    assert results["gsm8k"].score == 0.0
    assert "no run id submitted" in results["gsm8k"].reason
    assert coordinator.polled == ["r100"]  # no wasted call for the missing one


def test_verify_candidate_runs_ignores_runs_for_benchmarks_not_in_spec():
    """A miner submitting extra run ids cannot add benchmarks to the
    competition — only the spec's benchmarks are ever scored."""
    submission = make_submission(runs=[
        BenchmarkRun(b="mmlu", r="r100"),
        BenchmarkRun(b="not_in_spec", r="r999"),
    ])
    coordinator = StubCoordinator({"r100": make_status()})

    results = scorer.verify_candidate_runs(submission, make_spec(), coordinator)
    assert [r.benchmark for r in results] == ["mmlu"]
    assert "r999" not in coordinator.polled



# ---------------------------------------------------------------------------
# verify_all_candidates — persistence + aggregate
# ---------------------------------------------------------------------------

def test_verify_all_candidates_returns_scores_and_persists_rows():
    conn = make_db()
    submissions = {"hk1": make_submission()}
    coordinator = StubCoordinator({"r100": make_status()})

    scores = scorer.verify_all_candidates(conn, "comp1", submissions, make_spec(), coordinator)
    assert scores == {"hk1": {"mmlu": 0.8}}

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.8
    assert rows[0]["coordinator_run_id"] == "r100"


def test_verify_all_candidates_records_rejection_reason_and_actual_repo():
    """A rejected row must say why, and carry the repo the run really used —
    the dashboard reads these rows to explain a zero."""
    conn = make_db()
    submissions = {"hk1": make_submission()}
    coordinator = StubCoordinator({"r100": make_status(repo="someone/else")})

    scores = scorer.verify_all_candidates(conn, "comp1", submissions, make_spec(), coordinator)
    assert scores == {"hk1": {"mmlu": 0.0}}

    row = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")[0]
    assert row["status"] == "failed"
    assert row["score"] == 0.0
    assert "repo mismatch" in row["last_message"]
    assert row["repository"] == "someone/else"



# ---------------------------------------------------------------------------
# finalize_prechecked_candidate
# ---------------------------------------------------------------------------

def _insert_standby(conn, hotkey="hk1"):
    store.insert_revealed_candidate(
        conn, "comp1", hotkey, rank=0,
        submission_json=make_submission().model_dump_json(),
        reveal_block=5, status="prechecking",
    )
    conn.commit()


def test_finalize_scores_candidate_that_passes_floors():
    conn = make_db()
    _insert_standby(conn)

    scored = scorer.finalize_prechecked_candidate(
        conn, "comp1", "hk1", make_spec(), {"mmlu": 0.8}, measured_memory_kb=900,
    )
    assert scored is True
    assert store.get_candidate(conn, "comp1", "hk1")["status"] == "done"
    results = store.scoring_results_for_competition(conn, "comp1")
    assert len(results) == 1 and results[0]["hotkey"] == "hk1"


def test_finalize_fails_candidate_below_floor():
    """A candidate whose runs all scored 0.0 (every one rejected) reaches
    here and is rejected by the floors, exactly like a genuinely weak model."""
    conn = make_db()
    _insert_standby(conn)

    scored = scorer.finalize_prechecked_candidate(
        conn, "comp1", "hk1", make_spec(), {"mmlu": 0.0}, measured_memory_kb=900,
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(conn, "comp1") == []


def test_finalize_fails_candidate_over_memory_cap():
    conn = make_db()
    _insert_standby(conn)
    spec = make_spec(competition_type="ram_ceiling", max_memory_kb=1000)

    scored = scorer.finalize_prechecked_candidate(
        conn, "comp1", "hk1", spec, {"mmlu": 0.8}, measured_memory_kb=5000,
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "exceeded memory cap" in candidate["failure_reason"]


def test_finalize_benchmark_floor_ranks_by_measured_memory():
    """BENCHMARK_FLOOR scores the *measured* memory, not the claim — the
    self-reported value only orders the precheck queue."""
    conn = make_db()
    _insert_standby(conn)

    scorer.finalize_prechecked_candidate(
        conn, "comp1", "hk1", make_spec(), {"mmlu": 0.8}, measured_memory_kb=900,
    )
    result = store.scoring_results_for_competition(conn, "comp1")[0]
    assert result["final_score"] == -900.0
    assert result["max_memory_kb"] == 900


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
# precheck_one — unchanged by the rework, still the only ban path
# ---------------------------------------------------------------------------

def test_precheck_one_fails_when_repo_not_public(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: False)
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), FakeContainer(), make_db())
    assert result.passed is False
    assert "not publicly accessible" in result.reason



def test_precheck_one_passes_with_measured_ram(monkeypatch):
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class RamContainer:
        def check(self, repository, revision, filename, context_length):
            return PrecheckVerdict(provenance=None, sha256="a" * 64, ram=RamResult(passed=True, ram_bytes=1000 * 1024))

    result = scorer.precheck_one("hk1", make_submission(), make_spec(), RamContainer(), make_db())
    assert result.passed is True
    assert result.gguf_file == "model.gguf"
    assert result.measured_memory_kb == 1000


def test_precheck_one_allows_measured_ram_below_report(monkeypatch):
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class ConservativeRamContainer:
        def check(self, repository, revision, filename, context_length):
            return PrecheckVerdict(
                provenance=None,
                sha256="a" * 64,
                ram=RamResult(passed=True, ram_bytes=800 * 1024),
            )

    result = scorer.precheck_one(
        "hk1",
        make_submission(max_memory=1000),
        make_spec(),
        ConservativeRamContainer(),
        make_db(),
    )
    assert result.passed is True
    assert result.measured_memory_kb == 800


def test_precheck_one_bans_on_sha256_mismatch(monkeypatch):
    """The only remaining ban path. Score-lying is no longer possible, since
    scores come from the coordinator rather than the miner."""
    from competition.precheck_client import PrecheckVerdict

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class MismatchContainer:
        def check(self, repository, revision, filename, context_length):
            return PrecheckVerdict(provenance=None, ram=None, sha256="f" * 64)

    conn = make_db()
    result = scorer.precheck_one("hk1", make_submission(), make_spec(), MismatchContainer(), conn)
    assert result.passed is False
    assert "hotkey banned" in result.reason
    assert store.is_banned(conn, "hk1") is True


def test_precheck_one_disqualifies_on_max_memory_lie(monkeypatch):
    """max_memory stays self-reported in the commit (decision 4). A lie fails
    the candidate but does not ban — unlike a sha256 mismatch."""
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(scorer._hf_api, "list_repo_files", lambda repo_id, revision: ["model.gguf"])

    class LyingRamContainer:
        def check(self, repository, revision, filename, context_length):
            return PrecheckVerdict(provenance=None, sha256="a" * 64, ram=RamResult(passed=True, ram_bytes=5_000_000))

    conn = make_db()
    submission = make_submission(max_memory=1000)  # 1000 KB reported, ~4883 KB measured
    result = scorer.precheck_one("hk1", submission, make_spec(), LyingRamContainer(), conn)
    assert result.passed is False
    assert "max_memory lie" in result.reason
    assert store.is_banned(conn, "hk1") is False


def test_precheck_one_uses_submitted_filename_not_first_gguf(monkeypatch):
    """The submitted file must be checked, even when another .gguf sorts first."""
    from competition.precheck_client import PrecheckVerdict, RamResult

    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(
        scorer._hf_api, "list_repo_files",
        lambda repo_id, revision: ["aaa-decoy.gguf", "model.gguf"],
    )

    seen = {}

    class RecordingContainer:
        def check(self, repository, revision, filename, context_length):
            seen["filename"] = filename
            return PrecheckVerdict(provenance=None, sha256="a" * 64,
                                   ram=RamResult(passed=True, ram_bytes=1000 * 1024))

    result = scorer.precheck_one("hk1", make_submission(file="model.gguf"), make_spec(), RecordingContainer(), make_db())
    assert seen["filename"] == "model.gguf"
    assert result.passed is True


def test_precheck_one_fails_when_submitted_file_absent(monkeypatch):
    """A missing submitted file fails cleanly — no ban, no fallback."""
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
    monkeypatch.setattr(
        scorer._hf_api, "list_repo_files",
        lambda repo_id, revision: ["something-else.gguf"],
    )

    conn = make_db()
    result = scorer.precheck_one("hk1", make_submission(file="model.gguf"), make_spec(), FakeContainer(), conn)
    assert result.passed is False
    assert "not found at revision" in result.reason
    assert "something-else.gguf" in result.reason
    assert store.is_banned(conn, "hk1") is False

# ── KL comparison run verification ────────────────────────────────────────────
# Payload shapes are taken from a real fidelity-v1 run on the live coordinator
# (run 01f61c34, corpus-000002): a comparison run has no `model_source.model`,
# the candidate is nested under `.candidate.model`, and the base under `.base.model`.

KL_SPEC_FIELDS = dict(model_repo="Qwen/Qwen2.5-0.5B-Instruct", kl_corpus_id="corpus-000002")


def make_kl_status(**overrides) -> RunStatus:
    fields = dict(
        run_id="r1351", status=RunStatusCode.COMPLETED,
        repo="user/repo", revision="b" * 40,
        model_files=["model.gguf"], file_hashes={"model.gguf": ("sha256", "a" * 64)},
        base_repo="Qwen/Qwen2.5-0.5B-Instruct",
        dataset={"kind": "platform", "corpus_id": "corpus-000002"},
        comparison={
            "protocol": "fidelity-v1", "outcome": "complete", "reference": "transformers",
            "pairings": {scorer.KL_PAIRING: {"kl": {"mean": 0.0163, "quantiles": {"0.99": 0.11}}}},
        },
    )
    fields.update(overrides)
    return RunStatus(**fields)


def verify_kl(**overrides) -> scorer.RunVerification:
    return scorer.verify_kl_run(
        make_submission(), make_spec(**KL_SPEC_FIELDS), "r1351", make_kl_status(**overrides),
    )


def test_kl_run_verifies_and_returns_the_multiplier():
    result = verify_kl()
    assert result.ok and result.benchmark == "kl"
    assert 0.85 < result.score < 0.95


def test_kl_rejects_a_verifier_protocol_run():
    assert not verify_kl(comparison={"protocol": "verifier-v1", "score": 99}).ok


def test_kl_rejects_a_benchmark_run_with_no_comparison():
    assert not verify_kl(comparison=None).ok


def test_kl_rejects_an_invalid_candidate():
    comparison = dict(make_kl_status().comparison, outcome="candidate_invalid")
    assert not verify_kl(comparison=comparison).ok


def test_kl_rejects_a_gguf_reference_with_no_base_to_pin():
    comparison = dict(make_kl_status().comparison, reference="gguf")
    assert not verify_kl(comparison=comparison).ok


def test_kl_rejects_a_run_against_the_wrong_base():
    assert not verify_kl(base_repo="attacker/deliberately-weak").ok


def test_kl_rejects_a_miner_uploaded_dataset():
    assert not verify_kl(dataset={"kind": "upload", "dataset_id": "ds_1"}).ok


def test_kl_rejects_the_wrong_platform_corpus():
    assert not verify_kl(dataset={"kind": "platform", "corpus_id": "corpus-000099"}).ok


def test_kl_rejects_a_comparison_of_a_different_model():
    assert not verify_kl(repo="someone/else").ok
    assert not verify_kl(revision="f" * 40).ok
    assert not verify_kl(model_files=["other.gguf"]).ok
    assert not verify_kl(file_hashes={"model.gguf": ("sha256", "f" * 64)}).ok


def test_kl_skips_the_hash_check_for_non_sha256_algorithms():
    # HF serves xet/git_blob hashes for non-LFS files; they are not comparable.
    assert verify_kl(file_hashes={"model.gguf": ("xet", "zz")}).ok


def test_kl_rejects_a_missing_p99_rather_than_reading_it_as_zero():
    comparison = dict(
        make_kl_status().comparison,
        pairings={scorer.KL_PAIRING: {"kl": {"mean": 0.0163, "quantiles": {"0.9": 0.05}}}},
    )
    result = verify_kl(comparison=comparison)
    assert not result.ok and result.score == 0.0


def test_kl_rejects_the_wrong_pairing():
    comparison = dict(
        make_kl_status().comparison,
        pairings={"original_to_candidate": {"kl": {"quantiles": {"0.99": 0.11}}}},
    )
    assert not verify_kl(comparison=comparison).ok


def test_kl_rejects_a_run_that_is_not_finished():
    assert not verify_kl(status=RunStatusCode.RUNNING).ok
    assert not verify_kl(status=RunStatusCode.FAILED, failure_reason="pod died").ok


def test_every_kl_rejection_scores_zero_and_never_bans():
    for overrides in (
        {"comparison": None}, {"base_repo": "x/y"}, {"repo": "x/y"},
        {"dataset": {"kind": "upload", "dataset_id": "d"}}, {"status": RunStatusCode.RUNNING},
    ):
        result = verify_kl(**overrides)
        assert not result.ok and result.score == 0.0 and result.reason


def test_kl_run_is_skipped_when_the_competition_configures_no_corpus():
    results = scorer.verify_candidate_runs(
        make_submission(), make_spec(), _StubCoordinator(),
    )
    assert [r.benchmark for r in results] == ["mmlu"]


def test_missing_kl_run_id_is_reported_when_a_corpus_is_configured():
    results = scorer.verify_candidate_runs(
        make_submission(), make_spec(**KL_SPEC_FIELDS), _StubCoordinator(),
    )
    kl = [r for r in results if r.benchmark == "kl"]
    assert len(kl) == 1 and not kl[0].ok and kl[0].score == 0.0


class _StubCoordinator:
    def list_benchmarks(self):
        return {"mmlu"}

    def submit(self, *a, **k):
        raise NotImplementedError

    def poll(self, run_id):
        return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores={"mmlu": 0.8})
