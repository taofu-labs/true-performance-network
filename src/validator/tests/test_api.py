import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from validator import store
from validator import settings as validator_settings
from validator.api import LeaderApiMixin, _require_admin_auth


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
    app = web.Application()
    app.router.add_get("/v1/follower/scoring-results", v._h_follower_scoring_results)
    app.router.add_get("/v1/state/competitions/{competition_id}", v._h_state_competition)
    app.router.add_get("/v1/state/status", v._h_state_status)
    app.router.add_get("/v1/state/weights-history", v._h_state_weights_history)
    app.router.add_get("/v1/competitions", v._h_list_competitions)
    app.router.add_get("/v1/competitions/{competition_id}", v._h_get_competition)
    app.router.add_post("/v1/competitions", _require_admin_auth(v._h_upsert_competition))
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
        assert body == {"runs": [], "scored_status": None}


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
async def test_state_competition_returns_full_state_shape(conn):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/v1/state/competitions/comp1")
        body = await resp.json()
        assert body["competition_id"] == "comp1"
        assert body["runs"] == []


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
