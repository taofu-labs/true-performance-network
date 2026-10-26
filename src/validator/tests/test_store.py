import pathlib

import pytest
from pydantic import ValidationError

from common.models.competition import CompetitionSpec
from validator import store


def make_conn(tmp_path):
    return store.init_db(tmp_path / "validator.db")


def make_spec(comp_id="comp1", **overrides):
    fields = {
        "id": comp_id,
        "name": "Test Competition",
        "start_block": 0,
        "commit_end_block": 100,
        "scoring_end_block": 200,
        "emission_distribution": [1.0],
        "top_n": 1,
        "benchmarks": [{"name": "mmlu", "min_score": 0.5}],
    }
    fields.update(overrides)
    return CompetitionSpec.model_validate(fields)


def test_init_db_returns_same_connection_for_same_path(tmp_path):
    path = tmp_path / "validator.db"
    a = store.init_db(path)
    b = store.init_db(path)
    assert a is b


def test_is_scored_and_mark_scored(tmp_path):
    conn = make_conn(tmp_path)
    assert store.is_scored(conn, "comp1") is False
    store.mark_scored(conn, "comp1")
    assert store.is_scored(conn, "comp1") is True


def test_mark_scored_sets_matching_terminal_stage(tmp_path):
    conn = make_conn(tmp_path)
    store.mark_scored(conn, "comp1", status="failed_no_reveals")
    assert store.scored_status(conn, "comp1") == "failed_no_reveals"
    assert store.get_stage(conn, "comp1") == "failed_no_reveals"

    store.mark_scored(conn, "comp2", status="scored")
    assert store.get_stage(conn, "comp2") == "finalized"



def test_set_stage(tmp_path):
    conn = make_conn(tmp_path)
    store.set_stage(conn, "comp1", "stage2_scoring")
    assert store.get_stage(conn, "comp1") == "stage2_scoring"
    store.set_stage(conn, "comp1", "finalized")
    assert store.get_stage(conn, "comp1") == "finalized"


def test_bump_stage1_attempts_increments_and_caps(tmp_path):
    conn = make_conn(tmp_path)
    assert store.bump_stage1_attempts(conn, "comp1") == 1
    assert store.bump_stage1_attempts(conn, "comp1") == 2
    assert store.bump_stage1_attempts(conn, "comp1") == 3
    assert store.STAGE1_MAX_ATTEMPTS == 3



def test_record_and_get_latest_weights(tmp_path):
    conn = make_conn(tmp_path)
    assert store.latest_weights_for_competition(conn, "comp1") is None

    store.record_weights(conn, "comp1", {"hk1": 0.6})
    store.record_weights(conn, "comp1", {"hk1": 0.9})

    latest = store.latest_weights_for_competition(conn, "comp1")
    assert latest == {"hk1": 0.9}

    history = store.weights_history_for_competition(conn, "comp1")
    assert len(history) == 2


def test_upsert_and_get_competition(tmp_path):
    conn = make_conn(tmp_path)
    assert store.get_competition(conn, "comp1") is None

    store.upsert_competition(conn, make_spec())
    spec = store.get_competition(conn, "comp1")
    assert spec["id"] == "comp1"
    assert spec["name"] == "Test Competition"



def test_list_competitions(tmp_path):
    conn = make_conn(tmp_path)
    assert store.list_competitions(conn) == []

    store.upsert_competition(conn, make_spec("comp1"))
    store.upsert_competition(conn, make_spec("comp2"))

    ids = sorted(c["id"] for c in store.list_competitions(conn))
    assert ids == ["comp1", "comp2"]


# ---------------------------------------------------------------------------
# revealed_candidates (stage 1 output / stage 2 queue)
# ---------------------------------------------------------------------------

def test_insert_and_get_candidate(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "standby"
    assert candidate["rank"] == 0
    assert candidate["measured_memory_kb"] is None


def test_set_candidate_status(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    store.set_candidate_status(conn, "comp1", "hk1", "failed", "provenance fail")
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert candidate["failure_reason"] == "provenance fail"


def test_insert_revealed_candidates_persists_measured_memory(tmp_path):
    """The memory figure comes out of stage-1 verification, so it is written
    with the candidate rather than filled in by a later step."""
    conn = make_conn(tmp_path)
    store.insert_revealed_candidates(conn, "comp1", [{
        "hotkey": "hk1", "rank": 0, "submission_json": "{}", "reveal_block": 10,
        "status": "standby", "verified_scores": {"mmlu": 0.8}, "measured_memory_kb": 1234,
    }])
    assert store.get_candidate(conn, "comp1", "hk1")["measured_memory_kb"] == 1234


def test_insert_revealed_candidates_is_idempotent(tmp_path):
    """A scoring pass that dies partway leaves rows behind and the next tick
    re-runs the whole pass. A plain INSERT would raise on the primary key and
    burn the competition's stage-1 attempts."""
    conn = make_conn(tmp_path)
    row = dict(hotkey="hk1", rank=0, submission_json="{}", reveal_block=5,
               status="standby", verified_scores={"mmlu": 0.8}, measured_memory_kb=900)
    store.insert_revealed_candidates(conn, "comp1", [row])
    store.insert_revealed_candidates(
        conn, "comp1", [{**row, "status": "done", "measured_memory_kb": 950, "rank": 1}],
    )

    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "done"       # the re-run's values win
    assert candidate["measured_memory_kb"] == 950
    assert candidate["rank"] == 1
    assert len(store.all_candidates_for_competition(conn, "comp1")) == 1


def test_migration_rewrites_stored_competition_specs(tmp_path):
    """Stored specs are JSON and are validated on every leader tick, so one
    row still saying `ram_ceiling` would take down the whole loop, not just
    its own competition."""
    import json as _json
    import sqlite3

    from common.models.competition import CompetitionSpec

    path = tmp_path / "specs.db"
    old_db = sqlite3.connect(path)
    old_db.executescript("""
        CREATE TABLE competitions (id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                                   created_at REAL NOT NULL, updated_at REAL NOT NULL);
    """)
    stored = {
        "id": "old", "name": "old", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "top_n": 1, "emission_distribution": [1.0],
        "benchmarks": [{"name": "mmlu", "min_score": 0.5}],
        "competition_type": "ram_ceiling", "max_memory_kb": 4096,
        "ram_check_context_length": 4096,
    }
    old_db.execute("INSERT INTO competitions VALUES ('old', ?, 1.0, 1.0)", (_json.dumps(stored),))
    old_db.commit()
    old_db.close()

    conn = store.init_db(path)
    spec_json = conn.execute("SELECT spec_json FROM competitions WHERE id = 'old'").fetchone()[0]
    migrated = _json.loads(spec_json)
    assert migrated["competition_type"] == "gpu_memory_ceiling"
    assert migrated["max_gpu_memory_kb"] == 4096
    assert "max_memory_kb" not in migrated
    assert "ram_check_context_length" not in migrated
    # The whole point: it loads again.
    assert CompetitionSpec.model_validate(migrated).max_gpu_memory_kb == 4096


def test_migration_loads_a_real_production_spec(tmp_path):
    """Pinned against a spec captured from the live validator
    (val0.trueperformancenetwork.com, tpn-001). Every stored spec there is
    `ram_ceiling` with max_memory_kb and ram_check_context_length, and none of
    them load under the current model — so without this migration the leader
    raises on every tick, for every competition, not just the stale one.

    It also carries a `weight` key on each benchmark that BenchmarkTask does
    not define, which must keep being ignored rather than becoming an error.
    """
    import json as _json
    import sqlite3

    from common.models.competition import CompetitionSpec, CompetitionType

    raw = _json.loads(
        (pathlib.Path(__file__).parent / "fixtures" / "legacy_competition_spec.json").read_text()
    )
    assert raw["competition_type"] == "ram_ceiling"          # the captured shape
    with pytest.raises(ValidationError):
        CompetitionSpec.model_validate(raw)                  # ... does not load today

    path = tmp_path / "prod.db"
    old_db = sqlite3.connect(path)
    old_db.executescript("""
        CREATE TABLE competitions (id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                                   created_at REAL NOT NULL, updated_at REAL NOT NULL);
    """)
    old_db.execute("INSERT INTO competitions VALUES (?, ?, 1.0, 1.0)", (raw["id"], _json.dumps(raw)))
    old_db.commit()
    old_db.close()

    conn = store.init_db(path)
    migrated = _json.loads(
        conn.execute("SELECT spec_json FROM competitions WHERE id = ?", (raw["id"],)).fetchone()[0]
    )
    spec = CompetitionSpec.model_validate(migrated)
    assert spec.competition_type == CompetitionType.GPU_MEMORY_CEILING
    assert spec.max_gpu_memory_kb == raw["max_memory_kb"]    # the cap is carried over verbatim
    assert [t.name for t in spec.benchmarks] == [b["name"] for b in raw["benchmarks"]]


def test_migration_drops_an_unpinned_kl_corpus(tmp_path):
    """CompetitionSpec now refuses kl_corpus_id without model_repo, and stored
    specs are validated on every leader tick — so one such row would stall the
    whole loop. The corpus is dropped so the competition still loads and scores
    on its benchmarks, which is what it effectively did anyway: an unpinned base
    made the multiplier free for everyone."""
    import json as _json
    import sqlite3

    from common.models.competition import CompetitionSpec

    path = tmp_path / "unpinned.db"
    old_db = sqlite3.connect(path)
    old_db.executescript("""
        CREATE TABLE competitions (id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                                   created_at REAL NOT NULL, updated_at REAL NOT NULL);
    """)
    stored = {
        "id": "unpinned", "name": "unpinned", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "top_n": 1, "emission_distribution": [1.0],
        "benchmarks": [{"name": "mmlu", "min_score": 0.5}],
        "competition_type": "gpu_memory_ceiling", "max_gpu_memory_kb": 1000,
        "kl_corpus_id": "corpus-000002",   # no model_repo
    }
    old_db.execute("INSERT INTO competitions VALUES ('unpinned', ?, 1.0, 1.0)", (_json.dumps(stored),))
    old_db.commit()
    old_db.close()

    conn = store.init_db(path)
    migrated = _json.loads(
        conn.execute("SELECT spec_json FROM competitions WHERE id = 'unpinned'").fetchone()[0]
    )
    assert "kl_corpus_id" not in migrated
    # The whole point: the leader can load it again instead of raising every tick.
    assert CompetitionSpec.model_validate(migrated).kl_corpus_id is None


def test_migration_keeps_a_properly_pinned_kl_corpus(tmp_path):
    import json as _json
    import sqlite3

    path = tmp_path / "pinned.db"
    old_db = sqlite3.connect(path)
    old_db.executescript("""
        CREATE TABLE competitions (id TEXT PRIMARY KEY, spec_json TEXT NOT NULL,
                                   created_at REAL NOT NULL, updated_at REAL NOT NULL);
    """)
    stored = {
        "id": "pinned", "name": "pinned", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "top_n": 1, "emission_distribution": [1.0],
        "benchmarks": [{"name": "mmlu", "min_score": 0.5}],
        "competition_type": "gpu_memory_ceiling", "max_gpu_memory_kb": 1000,
        "kl_corpus_id": "corpus-000002", "model_repo": "Qwen/Qwen2.5-0.5B-Instruct",
    }
    old_db.execute("INSERT INTO competitions VALUES ('pinned', ?, 1.0, 1.0)", (_json.dumps(stored),))
    old_db.commit()
    old_db.close()

    conn = store.init_db(path)
    migrated = _json.loads(
        conn.execute("SELECT spec_json FROM competitions WHERE id = 'pinned'").fetchone()[0]
    )
    assert migrated["kl_corpus_id"] == "corpus-000002"


def test_migration_drops_banned_hotkeys_and_renames_memory_column(tmp_path):
    """An existing DB carrying the pre-removal schema must come forward without
    manual surgery: banning is gone and the memory column is a measurement."""
    import sqlite3

    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE banned_hotkeys (hotkey TEXT PRIMARY KEY, reason TEXT NOT NULL, banned_at REAL NOT NULL);
        CREATE TABLE scoring_results (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, final_score REAL NOT NULL,
            max_memory_kb INTEGER NOT NULL, finalized_at REAL NOT NULL,
            PRIMARY KEY (competition_id, hotkey));
    """)
    old.execute("INSERT INTO scoring_results VALUES ('comp1', 'hk1', 0.9, 4096, 1.0)")
    old.commit()
    old.close()

    conn = store.init_db(path)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "banned_hotkeys" not in tables
    row = conn.execute("SELECT * FROM scoring_results WHERE hotkey = 'hk1'").fetchone()
    assert row["measured_memory_kb"] == 4096  # the row survives the rename


def test_candidates_by_status_filters_and_orders_by_rank(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk2", rank=1, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk3", rank=2, submission_json="{}", reveal_block=10, status="failed", failure_reason="x")
    conn.commit()

    standby = store.candidates_by_status(conn, "comp1", ("standby",))
    assert [c["hotkey"] for c in standby] == ["hk1", "hk2"]



def test_all_candidates_for_competition_ordered_by_rank(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk2", rank=1, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    all_candidates = store.all_candidates_for_competition(conn, "comp1")
    assert [c["hotkey"] for c in all_candidates] == ["hk1", "hk2"]


# ---------------------------------------------------------------------------
# benchmark_results (run-id verification outcomes)
#
# The table keeps its full historical shape — the dashboard renders past
# competitions from these rows — so the lifecycle columns are still written,
# just with values that mean "verified" rather than "in progress".
# ---------------------------------------------------------------------------

def test_record_benchmark_verification_persists_a_verified_run(tmp_path):
    conn = make_conn(tmp_path)
    store.record_benchmark_verification(
        conn, "comp1", "hk1", "mmlu", run_id="r100",
        repository="user/repo", revision="a" * 40,
        status="completed", score=0.8,
    )

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.8
    assert rows[0]["coordinator_run_id"] == "r100"


def test_record_benchmark_verification_persists_rejection_with_reason(tmp_path):
    """A rejected run scores 0.0 and records why, plus the repo the run
    actually used — enough to explain the zero without the coordinator."""
    conn = make_conn(tmp_path)
    store.record_benchmark_verification(
        conn, "comp1", "hk1", "mmlu", run_id="r100",
        repository="someone/else", revision="b" * 40,
        status="failed", score=0.0, message="repo mismatch: ...",
    )

    row = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")[0]
    assert row["status"] == "failed"
    assert row["score"] == 0.0
    assert row["repository"] == "someone/else"
    assert "repo mismatch" in row["last_message"]



def test_record_benchmark_verification_is_idempotent_on_conflict(tmp_path):
    """Re-verifying (a retried stage 1) overwrites rather than duplicating."""
    conn = make_conn(tmp_path)
    store.record_benchmark_verification(
        conn, "comp1", "hk1", "mmlu", run_id="r100",
        repository="user/repo", revision="a" * 40, status="failed", score=0.0,
        message="run not complete at scoring time",
    )
    store.record_benchmark_verification(
        conn, "comp1", "hk1", "mmlu", run_id="r100",
        repository="user/repo", revision="a" * 40, status="completed", score=0.9,
    )

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.9


def test_migration_adds_verified_scores_and_leaves_other_tables_alone(tmp_path):
    """The rework migration is purely additive: one new column, and every
    existing table and row left exactly as it was."""
    import sqlite3

    db_path = tmp_path / "old.db"
    old = sqlite3.connect(db_path)
    old.executescript("""
        CREATE TABLE revealed_candidates (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, rank INTEGER NOT NULL,
            submission_json TEXT NOT NULL, reveal_block INTEGER NOT NULL, status TEXT NOT NULL,
            failure_reason TEXT, gguf_file TEXT, measured_memory_kb INTEGER,
            updated_at REAL NOT NULL, PRIMARY KEY (competition_id, hotkey));
        CREATE TABLE benchmark_results (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, benchmark_name TEXT NOT NULL,
            score REAL, coordinator_run_id TEXT NOT NULL, status TEXT NOT NULL,
            repository TEXT NOT NULL, revision TEXT NOT NULL, submitted_at REAL NOT NULL,
            updated_at REAL NOT NULL, phase TEXT, percent_complete REAL, last_message TEXT,
            PRIMARY KEY (competition_id, hotkey, benchmark_name));
        CREATE TABLE benchmark_runs (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, benchmark_name TEXT NOT NULL,
            repository TEXT NOT NULL, revision TEXT NOT NULL, coordinator_run_id TEXT NOT NULL,
            status TEXT NOT NULL, submitted_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (competition_id, hotkey, benchmark_name));
        INSERT INTO benchmark_results VALUES
            ('old','hk_hist','mmlu',0.66,'run-old','completed','user/old','c0ffee',
             1.0,2.0,'benchmarking',100.0,'done');
    """)
    old.commit()
    old.close()

    conn = store.init_db(db_path)

    candidate_columns = {r["name"] for r in conn.execute("PRAGMA table_info(revealed_candidates)")}
    assert "verified_scores_json" in candidate_columns

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "benchmark_runs" in tables  # superseded, but never dropped

    historical = store.benchmark_results_for_hotkey(conn, "old", "hk_hist")[0]
    assert historical["score"] == 0.66
    assert historical["phase"] == "benchmarking"
    assert historical["last_message"] == "done"


def test_migration_preserves_rows_in_superseded_benchmark_runs(tmp_path):
    """benchmark_runs was superseded by benchmark_results in the validator
    rewrite, which removed every accessor but left the table. An older
    validator did write to it, and those rows are competition history, so the
    migration never touches them."""
    import sqlite3

    db_path = tmp_path / "with_rows.db"
    old = sqlite3.connect(db_path)
    old.executescript("""
        CREATE TABLE revealed_candidates (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, rank INTEGER NOT NULL,
            submission_json TEXT NOT NULL, reveal_block INTEGER NOT NULL, status TEXT NOT NULL,
            failure_reason TEXT, gguf_file TEXT, measured_memory_kb INTEGER,
            updated_at REAL NOT NULL, PRIMARY KEY (competition_id, hotkey));
        CREATE TABLE benchmark_runs (
            competition_id TEXT NOT NULL, hotkey TEXT NOT NULL, benchmark_name TEXT NOT NULL,
            repository TEXT NOT NULL, revision TEXT NOT NULL, coordinator_run_id TEXT NOT NULL,
            status TEXT NOT NULL, submitted_at REAL NOT NULL, updated_at REAL NOT NULL,
            PRIMARY KEY (competition_id, hotkey, benchmark_name));
        INSERT INTO benchmark_runs VALUES
            ('ancient','hk_old','mmlu','user/old','c0ffee','run-ancient','completed',1.0,2.0);
    """)
    old.commit()
    old.close()

    conn = store.init_db(db_path)

    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "benchmark_runs" in tables
    surviving = conn.execute("SELECT coordinator_run_id FROM benchmark_runs").fetchone()
    assert surviving[0] == "run-ancient"


# ---------------------------------------------------------------------------
# scoring_results (stage 3 finalized happy-path results)
# ---------------------------------------------------------------------------

def test_record_and_read_scoring_results(tmp_path):
    conn = make_conn(tmp_path)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, measured_memory_kb=1000)
    store.record_scoring_result(conn, "comp1", "hk2", final_score=0.5, measured_memory_kb=2000)
    conn.commit()

    results = store.scoring_results_for_competition(conn, "comp1")
    assert [r["hotkey"] for r in results] == ["hk1", "hk2"]  # ordered by final_score DESC


def test_record_scoring_result_is_idempotent_on_conflict(tmp_path):
    conn = make_conn(tmp_path)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.5, measured_memory_kb=1000)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, measured_memory_kb=1000)
    conn.commit()

    results = store.scoring_results_for_competition(conn, "comp1")
    assert len(results) == 1
    assert results[0]["final_score"] == 0.9


def test_concurrent_writers_share_one_connection_safely(tmp_path):
    """Two threads driving the shared Connection must not lose or corrupt writes.

    Mirrors the validator's real shape: the event-loop thread and the single
    asyncio.to_thread worker both call store functions on the same handle.
    Without @_locked this trips sqlite3 "recursive use of cursors" /
    "InterfaceError" or silently drops rows.
    """
    import threading

    conn = make_conn(tmp_path)
    errors = []

    def writer(prefix):
        try:
            for i in range(100):
                store.insert_revealed_candidate(
                    conn, "comp1", f"{prefix}{i}", rank=i,
                    submission_json="{}", reveal_block=i, status="standby",
                )
                store.set_candidate_status(conn, "comp1", f"{prefix}{i}", "queued")
                store.candidates_by_status(conn, "comp1", ("standby", "queued"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(p,)) for p in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    rows = store.all_candidates_for_competition(conn, "comp1")
    assert len(rows) == 200
    assert all(r["status"] == "queued" for r in rows)


def test_pause_round_trip(tmp_path):
    conn = make_conn(tmp_path)
    assert store.is_paused(conn, "comp1") is False
    assert store.paused_at(conn, "comp1") is None

    store.set_paused(conn, "comp1", True)
    assert store.is_paused(conn, "comp1") is True
    assert store.paused_at(conn, "comp1") > 0

    store.set_paused(conn, "comp1", False)
    assert store.is_paused(conn, "comp1") is False
    assert store.paused_at(conn, "comp1") is None




def test_paused_at_column_migrates_onto_existing_db(tmp_path):
    """A DB created before paused_at existed gets the column added on open."""
    import sqlite3

    path = tmp_path / "validator.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(
        """
        CREATE TABLE scored_competitions (
            competition_id TEXT PRIMARY KEY,
            stage TEXT NOT NULL DEFAULT 'stage1_ranking',
            stage1_attempts INTEGER NOT NULL DEFAULT 0,
            scored_at REAL,
            status TEXT NOT NULL DEFAULT 'scoring'
        );
        """
    )
    legacy.execute(
        "INSERT INTO scored_competitions (competition_id, stage) VALUES ('comp1', 'stage2_scoring')"
    )
    legacy.commit()
    legacy.close()

    conn = store.init_db(path)
    assert store.get_stage(conn, "comp1") == "stage2_scoring"
    assert store.is_paused(conn, "comp1") is False
    store.set_paused(conn, "comp1", True)
    assert store.is_paused(conn, "comp1") is True


def test_scoring_results_break_ties_by_stage1_rank(tmp_path):
    """Results past top_n are stored, so a tie at the payout boundary decides
    who is paid. It must follow the stage-1 rank, not insertion order."""
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk_a", rank=0, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk_b", rank=1, submission_json="{}", reveal_block=10, status="standby")
    store.finalize_candidate(conn, "comp1", "hk_b", 0.5, 900)
    store.finalize_candidate(conn, "comp1", "hk_a", 0.5, 900)

    results = store.scoring_results_for_competition(conn, "comp1")
    assert [r["hotkey"] for r in results] == ["hk_a", "hk_b"]
