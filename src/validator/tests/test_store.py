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
        "benchmarks": [{"name": "mmlu", "min_score": 0.5, "weight": 1.0}],
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


def test_get_stage_defaults_to_stage1_for_unseen_competition(tmp_path):
    conn = make_conn(tmp_path)
    assert store.get_stage(conn, "never-seen") == "stage1_ranking"


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


def test_ban_and_is_banned(tmp_path):
    conn = make_conn(tmp_path)
    assert store.is_banned(conn, "hk1") is False
    store.ban(conn, "hk1", "cheating")
    assert store.is_banned(conn, "hk1") is True


def test_ban_is_idempotent(tmp_path):
    conn = make_conn(tmp_path)
    store.ban(conn, "hk1", "reason a")
    store.ban(conn, "hk1", "reason b")  # INSERT OR IGNORE — second call is a no-op
    assert store.is_banned(conn, "hk1") is True


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


def test_upsert_competition_updates_existing_row(tmp_path):
    conn = make_conn(tmp_path)
    store.upsert_competition(conn, make_spec())
    store.upsert_competition(conn, make_spec(name="Renamed"))

    assert store.get_competition(conn, "comp1")["name"] == "Renamed"
    assert len(store.list_competitions(conn)) == 1


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
    assert candidate["gguf_file"] is None
    assert candidate["measured_memory_kb"] is None


def test_set_candidate_status(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    store.set_candidate_status(conn, "comp1", "hk1", "failed", "provenance fail")
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "failed"
    assert candidate["failure_reason"] == "provenance fail"


def test_mark_precheck_passed_sets_queued_and_precheck_fields(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    store.mark_precheck_passed(conn, "comp1", "hk1", gguf_file="model.gguf", measured_memory_kb=1234)
    candidate = store.get_candidate(conn, "comp1", "hk1")
    assert candidate["status"] == "queued"
    assert candidate["gguf_file"] == "model.gguf"
    assert candidate["measured_memory_kb"] == 1234


def test_candidates_by_status_filters_and_orders_by_rank(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk2", rank=1, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk3", rank=2, submission_json="{}", reveal_block=10, status="failed", failure_reason="x")
    conn.commit()

    standby = store.candidates_by_status(conn, "comp1", ("standby",))
    assert [c["hotkey"] for c in standby] == ["hk1", "hk2"]


def test_count_candidates_by_status(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk2", rank=1, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    assert store.count_candidates_by_status(conn, "comp1", ("standby",)) == 2
    assert store.count_candidates_by_status(conn, "comp1", ("done",)) == 0


def test_all_candidates_for_competition_ordered_by_rank(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_revealed_candidate(conn, "comp1", "hk2", rank=1, submission_json="{}", reveal_block=10, status="standby")
    store.insert_revealed_candidate(conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=10, status="standby")
    conn.commit()

    all_candidates = store.all_candidates_for_competition(conn, "comp1")
    assert [c["hotkey"] for c in all_candidates] == ["hk1", "hk2"]


# ---------------------------------------------------------------------------
# benchmark_results (stage 2 benchmark persistence)
# ---------------------------------------------------------------------------

def test_insert_and_open_benchmark_results(tmp_path):
    conn = make_conn(tmp_path)
    assert store.open_benchmark_results(conn, "comp1") == []

    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    conn.commit()

    open_rows = store.open_benchmark_results(conn, "comp1")
    assert len(open_rows) == 1
    assert open_rows[0]["coordinator_run_id"] == "run-1"
    assert open_rows[0]["status"] == "submitted"
    assert open_rows[0]["score"] is None


def test_insert_benchmark_result_is_idempotent_on_conflict(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1-retry")
    conn.commit()

    open_rows = store.open_benchmark_results(conn, "comp1")
    assert len(open_rows) == 1
    assert open_rows[0]["coordinator_run_id"] == "run-1-retry"


def test_update_benchmark_result_persists_score_on_completion(tmp_path):
    """The C19 fix: a completed benchmark's score is persisted, not just its status."""
    conn = make_conn(tmp_path)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    conn.commit()

    store.update_benchmark_result(conn, "comp1", "hk1", "mmlu", "completed", score=0.85)
    conn.commit()

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["status"] == "completed"
    assert rows[0]["score"] == 0.85
    assert store.open_benchmark_results(conn, "comp1") == []  # no longer open


def test_update_benchmark_result_without_score_preserves_existing_score(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    conn.commit()
    store.update_benchmark_result(conn, "comp1", "hk1", "mmlu", "completed", score=0.5)
    conn.commit()

    # a later progress-only update (no score passed) must not clobber it
    store.update_benchmark_result(conn, "comp1", "hk1", "mmlu", "completed", phase="done")
    conn.commit()

    rows = store.benchmark_results_for_hotkey(conn, "comp1", "hk1")
    assert rows[0]["score"] == 0.5


def test_legacy_pending_resume_rows_are_migrated_to_submitted(tmp_path):
    """Rows left at the retired 'pending-resume' status by an older validator
    would otherwise be stranded: neither open nor terminal."""
    conn = make_conn(tmp_path)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    conn.execute("UPDATE benchmark_results SET status = 'pending-resume'")
    conn.commit()

    store._migrate_scored_competitions_stage_columns(conn)

    open_rows = store.open_benchmark_results(conn, "comp1")
    assert len(open_rows) == 1
    assert open_rows[0]["status"] == "submitted"


def test_open_benchmark_results_excludes_completed_and_failed(tmp_path):
    conn = make_conn(tmp_path)
    store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "user/repo", "a" * 40, "run-1")
    store.insert_benchmark_result(conn, "comp1", "hk2", "mmlu", "user/repo", "a" * 40, "run-2")
    conn.commit()
    store.update_benchmark_result(conn, "comp1", "hk1", "mmlu", "completed", score=0.5)
    store.update_benchmark_result(conn, "comp1", "hk2", "mmlu", "failed")
    conn.commit()

    assert store.open_benchmark_results(conn, "comp1") == []


# ---------------------------------------------------------------------------
# scoring_results (stage 3 finalized happy-path results)
# ---------------------------------------------------------------------------

def test_record_and_read_scoring_results(tmp_path):
    conn = make_conn(tmp_path)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, max_memory_kb=1000)
    store.record_scoring_result(conn, "comp1", "hk2", final_score=0.5, max_memory_kb=2000)
    conn.commit()

    results = store.scoring_results_for_competition(conn, "comp1")
    assert [r["hotkey"] for r in results] == ["hk1", "hk2"]  # ordered by final_score DESC


def test_record_scoring_result_is_idempotent_on_conflict(tmp_path):
    conn = make_conn(tmp_path)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.5, max_memory_kb=1000)
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, max_memory_kb=1000)
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
