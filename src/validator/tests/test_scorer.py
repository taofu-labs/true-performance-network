import json
import pathlib
import sqlite3

import pytest

from common.models.competition import BenchmarkTask, CompetitionSpec
from common.models.submission import BenchmarkRun, MinerSubmission
from competition.benchmark_client import RunStatus, RunStatusCode, _extract_model_identity
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
        measured_memory_bytes=900 * 1024,
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

    verified = scorer.verify_all_candidates(conn, "comp1", submissions, make_spec(), coordinator)
    assert verified["hk1"].scores == {"mmlu": 0.8}
    # A benchmark run carries its suite-level figure, so a competition with no
    # KL check still has a memory number to enforce a cap against.
    assert verified["hk1"].measured_memory_kb == 900

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

    verified = scorer.verify_all_candidates(conn, "comp1", submissions, make_spec(), coordinator)
    assert verified["hk1"].scores == {"mmlu": 0.0}
    # A rejected run's memory describes some other model, so it is not carried.
    assert verified["hk1"].measured_memory_kb == 0

    row = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")[0]
    assert row["status"] == "failed"
    assert row["score"] == 0.0
    assert "repo mismatch" in row["last_message"]
    assert row["repository"] == "someone/else"



# ---------------------------------------------------------------------------
# finalize_candidate
# ---------------------------------------------------------------------------

def _insert_standby(conn, hotkey="hk1"):
    store.insert_revealed_candidate(
        conn, "comp1", hotkey, rank=0,
        submission_json=make_submission().model_dump_json(),
        reveal_block=5, status="standby",
    )
    conn.commit()


def verification(scores, memory_kb=900) -> scorer.CandidateVerification:
    return scorer.CandidateVerification(scores=scores, measured_memory_kb=memory_kb)


def test_finalize_scores_candidate_that_passes_floors():
    conn = make_db()
    _insert_standby(conn)

    scored = scorer.finalize_candidate(
        conn, "comp1", "hk1", make_spec(), verification({"mmlu": 0.8}),
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

    scored = scorer.finalize_candidate(
        conn, "comp1", "hk1", make_spec(), verification({"mmlu": 0.0}),
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "failed floors" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(conn, "comp1") == []


def test_finalize_fails_candidate_over_memory_cap():
    conn = make_db()
    _insert_standby(conn)
    spec = make_spec(competition_type="gpu_memory_ceiling", max_gpu_memory_kb=1000)

    scored = scorer.finalize_candidate(
        conn, "comp1", "hk1", spec, verification({"mmlu": 0.8}, memory_kb=5000),
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "exceeded memory cap" in candidate["failure_reason"]


def test_finalize_benchmark_floor_ranks_by_measured_memory():
    """BENCHMARK_FLOOR scores the memory the coordinator measured. Miners no
    longer self-report one, so there is nothing to rank on but the measurement."""
    conn = make_db()
    _insert_standby(conn)

    scorer.finalize_candidate(
        conn, "comp1", "hk1", make_spec(), verification({"mmlu": 0.8}),
    )
    result = store.scoring_results_for_competition(conn, "comp1")[0]
    assert result["final_score"] == -900.0
    assert result["measured_memory_kb"] == 900


def test_finalize_refuses_to_pay_a_zero_scoring_candidate():
    """Emission shares go by rank position, so a candidate scoring 0 — most
    often one whose KL run could not be verified — must not occupy a payable
    slot. In a thin field it would otherwise draw a share."""
    conn = make_db()
    _insert_standby(conn)
    spec = make_spec(competition_type="gpu_memory_ceiling", max_gpu_memory_kb=10_000,
                     **KL_SPEC_FIELDS)

    scored = scorer.finalize_candidate(
        conn, "comp1", "hk1", spec, verification({"mmlu": 0.8, "kl": 0.0}),
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "nothing payable" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(conn, "comp1") == []


def test_finalize_still_pays_benchmark_floor_at_zero_benchmarks():
    """BENCHMARK_FLOOR scores negated memory, so its score is never 0 and the
    zero guard must not touch it — the floors already rejected weak models."""
    conn = make_db()
    _insert_standby(conn)

    assert scorer.finalize_candidate(
        conn, "comp1", "hk1", make_spec(), verification({"mmlu": 0.8}),
    ) is True
    assert store.scoring_results_for_competition(conn, "comp1")[0]["final_score"] == -900.0


def test_verify_run_rejects_a_private_repo():
    """A winner's model must be downloadable. The validator never contacts the
    Hub, so the run's own view of the repo is the only evidence."""
    result = scorer.verify_run(
        make_submission(), "mmlu", "r100", make_status(repo_private=True),
    )
    assert result.ok is False
    assert "private" in result.reason


def test_verify_run_rejects_a_gated_repo():
    result = scorer.verify_run(
        make_submission(), "mmlu", "r100", make_status(repo_gated=True),
    )
    assert result.ok is False
    assert "gated" in result.reason


def test_kl_run_rejects_a_private_repo():
    assert verify_kl(repo_private=True).ok is False
    assert "private" in verify_kl(repo_private=True).reason


def test_finalize_fails_candidate_whose_memory_is_unknown():
    """Scoring 0 KB would pass every cap and, under BENCHMARK_FLOOR, win
    outright. A candidate with no measurement is failed instead."""
    conn = make_db()
    _insert_standby(conn)

    scored = scorer.finalize_candidate(
        conn, "comp1", "hk1", make_spec(), verification({"mmlu": 0.8}, memory_kb=0),
    )
    assert scored is False
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert "GPU memory" in candidate["failure_reason"]
    assert store.scoring_results_for_competition(conn, "comp1") == []


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


# ── KL comparison run verification ────────────────────────────────────────────
# Payload shapes come from a real fidelity-v1 run on the live coordinator
# (run 01f61c34, corpus-000002, Q4_K_M: P99 0.1455 nats -> x0.8646).

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
            "pairings": {scorer.KL_PAIRING: {"kl": {"mean": 0.0247, "quantiles": {"0.99": 0.1455}}}},
        },
        measured_memory_bytes=1_700_000_000,
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
    assert result.score == pytest.approx(0.8646, abs=1e-4)


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
    assert verify_kl(file_hashes={"model.gguf": ("xet", "zz")}).ok


def test_kl_rejects_a_missing_p99_rather_than_reading_it_as_zero():
    comparison = dict(
        make_kl_status().comparison,
        pairings={scorer.KL_PAIRING: {"kl": {"mean": 0.0247, "quantiles": {"0.9": 0.05}}}},
    )
    result = verify_kl(comparison=comparison)
    assert not result.ok and result.score == 0.0


def test_kl_rejects_the_wrong_pairing():
    comparison = dict(
        make_kl_status().comparison,
        pairings={"original_to_candidate": {"kl": {"quantiles": {"0.99": 0.1455}}}},
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


# ── Live coordinator payload ──────────────────────────────────────────────────

def _live_status() -> dict:
    path = pathlib.Path(__file__).parent / "fixtures" / "comparison_status.json"
    return json.loads(path.read_text())


def test_parses_a_real_comparison_status_payload():
    """Guards the nesting against coordinator drift: a comparison run carries no
    model_source.model, so the candidate must be read from .candidate.model."""
    identity = _extract_model_identity(_live_status())
    assert identity["repo"] == "bartowski/Qwen2.5-0.5B-Instruct-GGUF"
    assert identity["revision"] == "41ba88dbac95fed2528c92514c131d73eb5a174b"
    assert identity["model_files"] == ["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"]
    assert identity["file_hashes"]["Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"][0] == "sha256"
    assert identity["base_repo"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert identity["dataset"]["corpus_id"] == "corpus-000002"
    assert identity["comparison"]["protocol"] == "fidelity-v1"


def test_verifies_a_real_comparison_run_end_to_end():
    data = _live_status()
    status = RunStatus(
        run_id=data["run_id"], status=RunStatusCode.COMPLETED, **_extract_model_identity(data),
    )
    candidate = data["model_source"]["candidate"]["model"]
    submission = make_submission(
        repository=candidate["repo"], file=candidate["files"][0]["path"],
        file_sha256=candidate["files"][0]["hash"]["value"],
        huggingface_revision=candidate["sha"],
        runs=[BenchmarkRun(b="kl", r=data["short_id"])],
    )
    spec = make_spec(
        model_repo=data["model_source"]["base"]["model"]["repo"],
        kl_corpus_id=data["model_source"]["dataset"]["corpus_id"],
    )

    # This capture predates the coordinator's `vram` block (benchmark server
    # 4.59.0), so it has no measured memory and is therefore unscoreable — which
    # is exactly the right outcome: the cap cannot be enforced without a figure.
    result = scorer.verify_kl_run(submission, spec, data["short_id"], status)
    assert result.ok is False
    assert "no measured GPU memory" in result.reason

    # With the measurement present, the same run verifies and prices the KL.
    identity = _extract_model_identity(data)
    assert identity["measured_memory_bytes"] is None  # no vram block in this capture
    measured = RunStatus(
        run_id=data["run_id"], status=RunStatusCode.COMPLETED,
        **{**identity, "measured_memory_bytes": 1_700_000_000},
    )
    priced = scorer.verify_kl_run(submission, spec, data["short_id"], measured)
    assert priced.ok, priced.reason
    assert priced.score == pytest.approx(0.864555, abs=1e-6)
    assert priced.measured_memory_kb == 1_700_000_000 // 1024

    # The same run against a competition on a different base earns nothing.
    other_base = spec.model_copy(update={"model_repo": "meta-llama/Llama-3.1-8B"})
    assert scorer.verify_kl_run(submission, other_base, data["short_id"], measured).score == 0.0


def test_parses_the_vram_block_a_comparison_reports():
    """Synthetic, not a capture: the fixture run predates benchmark server
    4.59.0. Shape per src/vram.js — the candidate's role is the committed model,
    and `estimated` is theory from the quote, never a measurement."""
    data = _live_status()
    data["comparison"]["models"]["candidate"]["vram"] = {
        "estimated": {"total_bytes": 999_999_999},
        "measured": {
            "phase": "after_load", "runtime": "llama_cpp",
            "method": "torch_mem_get_info_delta", "scope": "all_visible_gpus",
            "gpu_count": 1, "total_bytes": 1_700_000_000,
            "per_gpu_bytes": [1_700_000_000],
        },
    }
    data["comparison"]["models"]["base"]["vram"] = {
        "estimated": None,
        "measured": {
            "runtime": "transformers", "method": "torch_mem_get_info_delta",
            "total_bytes": 9_000_000_000,
        },
    }
    identity = _extract_model_identity(data)
    assert identity["measured_memory_bytes"] == 1_700_000_000
