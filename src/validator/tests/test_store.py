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


def test_record_scoring_run_and_scoring_results_since(tmp_path):
    from validator.scorer import OutcomeKind, ScoringOutcome
    from common.models.submission import MinerSubmission, ScoringResult, Claim

    conn = make_conn(tmp_path)
    submission = MinerSubmission(
        competition_id="comp1",
        claims=[Claim(b="mmlu", s=0.7)],
        repository="user/repo",
        file="model.gguf",
        file_sha256="a" * 64,
        max_memory=1000,
        huggingface_revision="a" * 40,
    )
    outcome = ScoringOutcome(
        kind=OutcomeKind.SCORED,
        result=ScoringResult(
            hotkey="hk1", competition_id="comp1", passed_floors=True, disqualified=False,
            actual_scores={"mmlu": 0.7}, final_score=0.7, max_memory_kb=1000,
            lying_detected=False, eval_backend="coordinator",
        ),
    )

    run_id = store.record_scoring_run(conn, "comp1", block=100, outcomes=[outcome], reveals={"hk1": submission})
    assert run_id == 1

    runs = store.scoring_results_since(conn, "comp1")
    assert len(runs) == 1
    assert runs[0]["results"][0]["hotkey"] == "hk1"
    assert runs[0]["results"][0]["final_score"] == 0.7

    assert store.scoring_results_since(conn, "comp1", after_run_id=run_id) == []


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
