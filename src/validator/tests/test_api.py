import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from validator import store
from validator import settings as validator_settings
from validator.api import MAX_REQUEST_BODY_BYTES, LeaderApiMixin, _require_admin_auth


class FakeSubtensor:
    def block(self):
        return 500


class FakeValidator(LeaderApiMixin):
    def __init__(self, conn):
        self._db = conn
        self.subtensor = FakeSubtensor()

    async def get_validator_status(self):
        return {"available": True}


def make_app(conn):
    v = FakeValidator(conn)
    app = web.Application(client_max_size=MAX_REQUEST_BODY_BYTES)
    app.router.add_get("/v1/follower/scoring-results", v._h_follower_scoring_results)
    app.router.add_get("/v1/state/competitions/{competition_id}", v._h_state_competition)
    app.router.add_get("/v1/state/status", v._h_state_status)
    app.router.add_get("/v1/state/weights-history", v._h_state_weights_history)
    app.router.add_get("/v1/competitions", v._h_list_competitions)
    app.router.add_get("/v1/competitions/{competition_id}", v._h_get_competition)
    app.router.add_post("/v1/competitions", _require_admin_auth(v._h_upsert_competition))
    app.router.add_delete("/v1/competitions/{competition_id}", _require_admin_auth(v._h_delete_competition))
    app.router.add_post(
        "/v1/competitions/{competition_id}/reset-scoring",
        _require_admin_auth(v._h_reset_competition_scoring),
    )
    return app


def make_spec_json(comp_id="comp1", **overrides):
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
    return fields


@pytest.fixture
def conn(tmp_path):
    return store.init_db(tmp_path / "validator.db")


@pytest.mark.asyncio
async def test_follower_scoring_results_requires_competition_id(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/follower/scoring-results")
        assert resp.status == 400


@pytest.mark.asyncio
async def test_follower_scoring_results_empty_for_unknown_competition(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/follower/scoring-results", params={"competition_id": "comp1"})
        assert resp.status == 200
        body = await resp.json()
        assert body == {"results": [], "scored_status": None}


@pytest.mark.asyncio
async def test_follower_scoring_results_hidden_until_finalized(conn):
    """A competition mid-stage2_scoring must not leak partial results to
    followers — only stage == 'finalized' exposes anything here."""
    store.record_scoring_result(conn, "comp1", "hk1", final_score=0.7, max_memory_kb=1000)
    conn.commit()
    store.set_stage(conn, "comp1", "stage2_scoring")

    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/follower/scoring-results", params={"competition_id": "comp1"})
        body = await resp.json()
        assert body["results"] == []

    store.set_stage(conn, "comp1", "finalized")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/follower/scoring-results", params={"competition_id": "comp1"})
        body = await resp.json()
        assert body["results"] == [{"competition_id": "comp1", "hotkey": "hk1", "final_score": 0.7,
                                     "max_memory_kb": 1000, "finalized_at": body["results"][0]["finalized_at"]}]


@pytest.mark.asyncio
async def test_state_weights_history_requires_competition_id(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/state/weights-history")
        assert resp.status == 400


@pytest.mark.asyncio
async def test_state_weights_history_returns_recorded_weights(conn):
    store.record_weights(conn, "comp1", {"hk1": 0.5})
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/state/weights-history", params={"competition_id": "comp1"})
        body = await resp.json()
        assert len(body["weights_history"]) == 1
        assert body["weights_history"][0]["weights"] == {"hk1": 0.5}


@pytest.mark.asyncio
async def test_state_status_returns_validator_status(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/state/status")
        body = await resp.json()
        assert body == {"available": True}


@pytest.mark.asyncio
async def test_state_competition_returns_full_state_shape(conn, monkeypatch):
    """Full state shape for a competition that exists but has no activity yet."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        resp = await client.get("/v1/state/competitions/comp1")
        body = await resp.json()
        assert body["competition_id"] == "comp1"
        assert body["candidates"] == []
        assert body["benchmark_results"] == []
        assert body["results"] == []


@pytest.mark.asyncio
async def test_list_competitions_empty(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/competitions")
        assert resp.status == 200
        body = await resp.json()
        assert body == {"competitions": []}


@pytest.mark.asyncio
async def test_get_competition_not_found(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/competitions/missing")
        assert resp.status == 404


@pytest.mark.asyncio
async def test_upsert_competition_requires_auth(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post("/v1/competitions", json=make_spec_json())
        assert resp.status == 401


@pytest.mark.asyncio
async def test_upsert_competition_rejects_wrong_key(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post(
            "/v1/competitions",
            json=make_spec_json(),
            headers={"Authorization": "Bearer wrong-key"},
        )
        assert resp.status == 401


@pytest.mark.asyncio
async def test_upsert_competition_503_when_key_unset(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", None)
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post(
            "/v1/competitions",
            json=make_spec_json(),
            headers={"Authorization": "Bearer anything"},
        )
        assert resp.status == 503


@pytest.mark.asyncio
async def test_upsert_competition_rejects_invalid_spec(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post(
            "/v1/competitions",
            json={"id": "bad"},  # missing required fields
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status == 422


@pytest.mark.asyncio
async def test_upsert_competition_success_then_listed(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post(
            "/v1/competitions",
            json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status == 200

        resp = await client.get("/v1/competitions/comp1")
        assert resp.status == 200
        body = await resp.json()
        assert body["id"] == "comp1"

        resp = await client.get("/v1/competitions")
        body = await resp.json()
        assert len(body["competitions"]) == 1


@pytest.mark.asyncio
async def test_list_competitions_active_filter(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions",
            json=make_spec_json("past", start_block=0, commit_end_block=10, scoring_end_block=20),
            headers={"Authorization": "Bearer secret-key"},
        )
        await client.post(
            "/v1/competitions",
            json=make_spec_json("current", start_block=0, commit_end_block=1000, scoring_end_block=2000),
            headers={"Authorization": "Bearer secret-key"},
        )

        # FakeSubtensor.block() == 500, "past" already ended distribution, "current" is OPEN
        resp = await client.get("/v1/competitions", params={"active": "true"})
        body = await resp.json()
        ids = [c["id"] for c in body["competitions"]]
        assert ids == ["current"]


@pytest.mark.asyncio
async def test_state_competition_unknown_id_returns_404(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/state/competitions/does-not-exist")
        assert resp.status == 404
        assert (await resp.json())["error"] == "competition not found"


@pytest.mark.asyncio
async def test_state_competition_known_id_still_returns_state(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        resp = await client.get("/v1/state/competitions/comp1")
        assert resp.status == 200
        assert (await resp.json())["competition_id"] == "comp1"


@pytest.mark.asyncio
async def test_delete_competition_requires_auth(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        resp = await client.delete("/v1/competitions/comp1")
        assert resp.status == 401
        assert (await (await client.get("/v1/competitions/comp1")).json())["id"] == "comp1"


@pytest.mark.asyncio
async def test_delete_competition_unknown_id_returns_404(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.delete(
            "/v1/competitions/nope", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 404


@pytest.mark.asyncio
async def test_delete_competition_removes_clean_spec(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json("typo-id"),
            headers={"Authorization": "Bearer secret-key"},
        )
        resp = await client.delete(
            "/v1/competitions/typo-id", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        assert (await resp.json())["deleted"] == "typo-id"

        assert (await client.get("/v1/competitions/typo-id")).status == 404
        body = await (await client.get("/v1/competitions")).json()
        assert body["competitions"] == []


@pytest.mark.asyncio
async def test_delete_competition_refuses_when_scoring_state_exists(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, max_memory_kb=100)
        conn.commit()

        resp = await client.delete(
            "/v1/competitions/comp1", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        assert (await client.get("/v1/competitions/comp1")).status == 200


@pytest.mark.asyncio
async def test_delete_competition_force_overrides_state_guard(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, max_memory_kb=100)
        conn.commit()

        resp = await client.delete(
            "/v1/competitions/comp1?force=true",
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status == 200
        assert (await resp.json())["forced"] is True
        assert (await client.get("/v1/competitions/comp1")).status == 404


@pytest.mark.asyncio
async def test_oversized_body_rejected(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        spec = make_spec_json()
        spec["description"] = "x" * (MAX_REQUEST_BODY_BYTES + 1)
        resp = await client.post(
            "/v1/competitions", json=spec,
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status == 413


@pytest.mark.asyncio
async def test_list_competitions_rejects_non_integer_block(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/competitions", params={"active": "true", "block": "abc"})
        assert resp.status == 400
        assert "must be an integer" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_list_competitions_rejects_negative_block(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/competitions", params={"active": "true", "block": "-5"})
        assert resp.status == 400
        assert "non-negative" in (await resp.json())["error"]


@pytest.mark.asyncio
async def test_list_competitions_accepts_valid_block(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions",
            json=make_spec_json("c", start_block=0, commit_end_block=1000, scoring_end_block=2000),
            headers={"Authorization": "Bearer secret-key"},
        )
        resp = await client.get("/v1/competitions", params={"active": "true", "block": "500"})
        assert resp.status == 200
        assert [c["id"] for c in (await resp.json())["competitions"]] == ["c"]


async def _seed_open_competition(client, comp_id="comp1"):
    """Scoring window still open at FakeSubtensor block 500."""
    await client.post(
        "/v1/competitions",
        json=make_spec_json(comp_id, start_block=0, commit_end_block=1000, scoring_end_block=2000),
        headers={"Authorization": "Bearer secret-key"},
    )


@pytest.mark.asyncio
async def test_reset_scoring_requires_auth(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        resp = await client.post("/v1/competitions/comp1/reset-scoring")
        assert resp.status == 401


@pytest.mark.asyncio
async def test_reset_scoring_unknown_competition_404(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post(
            "/v1/competitions/ghost/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 404


@pytest.mark.asyncio
async def test_reset_scoring_recovers_infra_failure(conn, monkeypatch):
    """failed_stage1_infra with the window still open — the case this exists for."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)

        store.insert_revealed_candidate(
            conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=5, status="failed"
        )
        store.insert_benchmark_result(conn, "comp1", "hk1", "mmlu", "r", "rev", "run1")
        store.record_scoring_result(conn, "comp1", "hk1", final_score=0.5, max_memory_kb=10)
        conn.commit()
        store.mark_scored(conn, "comp1", status="failed_stage1_infra")
        assert store.is_scored(conn, "comp1") is True

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["previous_status"] == "failed_stage1_infra"
        assert body["deleted"]["revealed_candidates"] == 1
        assert body["deleted"]["scoring_results"] == 1

        assert store.is_scored(conn, "comp1") is False
        assert store.scored_status(conn, "comp1") is None
        assert store.get_stage(conn, "comp1") == "stage1_ranking"
        assert store.all_candidates_for_competition(conn, "comp1") == []
        assert store.scoring_results_for_competition(conn, "comp1") == []
        assert store.get_competition(conn, "comp1") is not None


@pytest.mark.asyncio
async def test_reset_scoring_preserves_bans_and_weights_history(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        store.ban(conn, "cheater", "sha256 mismatch")
        store.record_weights(conn, "comp1", {"hk1": 1.0})
        store.mark_scored(conn, "comp1", status="failed_stage1_infra")

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200

        assert store.is_banned(conn, "cheater") is True
        assert len(store.weights_history_for_competition(conn, "comp1")) == 1


@pytest.mark.asyncio
async def test_reset_scoring_refuses_successfully_scored(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        store.mark_scored(conn, "comp1", status="scored")

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        assert store.is_scored(conn, "comp1") is True


@pytest.mark.asyncio
async def test_reset_scoring_refuses_stage2(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        store.insert_revealed_candidate(
            conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=5, status="benchmarking"
        )
        conn.commit()
        store.set_stage(conn, "comp1", "stage2_scoring")

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        assert (await resp.json())["stage"] == "stage2_scoring"
        assert store.get_stage(conn, "comp1") == "stage2_scoring"
        assert len(store.all_candidates_for_competition(conn, "comp1")) == 1


@pytest.mark.asyncio
async def test_reset_scoring_refuses_finalized(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        store.record_scoring_result(conn, "comp1", "hk1", final_score=0.9, max_memory_kb=10)
        conn.commit()
        store.mark_scored(conn, "comp1", status="scored")
        assert store.get_stage(conn, "comp1") == "finalized"

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        assert store.is_scored(conn, "comp1") is True
        assert len(store.scoring_results_for_competition(conn, "comp1")) == 1


@pytest.mark.asyncio
async def test_reset_scoring_resets_untouched_stage1_competition(conn, monkeypatch):
    """No scored_competitions row yet — get_stage() defaults to stage1_ranking."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["previous_stage"] == "stage1_ranking"
        assert body["previous_status"] is None
        assert all(n == 0 for n in body["deleted"].values())


@pytest.mark.asyncio
async def test_reset_scoring_ignores_force_query_param(conn, monkeypatch):
    """force was removed — a closed window is refused regardless."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        store.mark_scored(conn, "comp1", status="failed_stage1_infra")

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring?force=true",
            headers={"Authorization": "Bearer secret-key"},
        )
        assert resp.status == 409
        assert store.is_scored(conn, "comp1") is True


@pytest.mark.asyncio
async def test_reset_scoring_refuses_when_window_closed(conn, monkeypatch):
    """FakeSubtensor is at block 500; the default spec ends scoring at 200."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await client.post(
            "/v1/competitions", json=make_spec_json(),
            headers={"Authorization": "Bearer secret-key"},
        )
        store.mark_scored(conn, "comp1", status="failed_stage1_infra")

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        body = await resp.json()
        assert body["current_block"] == 500
        assert body["scoring_end_block"] == 200
        assert store.is_scored(conn, "comp1") is True
