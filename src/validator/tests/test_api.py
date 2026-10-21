import json
import time

import pytest
from bittensor import http_auth, sp_core
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from common.models.competition import BenchmarkTask, CompetitionSpec, CompetitionType
from validator import store
from validator import settings as validator_settings
from validator import api as api_module
from validator.api import (
    MAX_REQUEST_BODY_BYTES, LeaderApiMixin, creation_errors, _require_admin_auth,
    _require_admin_signer,
)


def _keypair(mnemonic):
    return sp_core.Keypair.create_from_mnemonic(mnemonic, sp_core.CRYPTO_SR25519)


LEADER = _keypair(sp_core.Keypair.generate_mnemonic(12))
ADMIN = _keypair(sp_core.Keypair.generate_mnemonic(12))


class FakeSubtensor:
    def block(self):
        return 500


class FakeValidator(LeaderApiMixin):
    def __init__(self, conn):
        self._db = conn
        self.subtensor = FakeSubtensor()
        self.hotkey = LEADER.ss58_address

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
    app.router.add_post(
        "/v1/competitions/{competition_id}/pause", _require_admin_auth(v._h_pause_competition)
    )
    app.router.add_post(
        "/v1/competitions/{competition_id}/resume", _require_admin_auth(v._h_resume_competition)
    )
    app.router.add_get("/admin", v._h_admin_page)
    app.router.add_get("/v1/admin/head", v._h_admin_head)
    app.router.add_post(
        "/v1/admin/competitions", _require_admin_signer(v._h_create_competition, v.hotkey)
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
        "benchmarks": [{"name": "mmlu", "min_score": 0.5}],
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




async def _seed_open_competition(client, comp_id="comp1"):
    """Scoring window still open at FakeSubtensor block 500."""
    await client.post(
        "/v1/competitions",
        json=make_spec_json(comp_id, start_block=0, commit_end_block=1000, scoring_end_block=2000),
        headers={"Authorization": "Bearer secret-key"},
    )




@pytest.mark.asyncio
async def test_reset_scoring_recovers_infra_failure(conn, monkeypatch):
    """failed_stage1_infra with the window still open — the case this exists for."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)

        store.insert_revealed_candidate(
            conn, "comp1", "hk1", rank=0, submission_json="{}", reveal_block=5, status="failed"
        )
        store.record_benchmark_verification(
            conn, "comp1", "hk1", "mmlu", run_id="r100", repository="user/repo",
            revision="rev", status="completed", score=0.5,
        )
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


# ── pause / resume ───────────────────────────────────────────────────




@pytest.mark.asyncio
async def test_pause_sets_flag_and_is_idempotent(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)

        resp = await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["paused"] is True
        assert body["paused_at"] is not None
        assert store.is_paused(conn, "comp1") is True

        # re-pausing is a 200, not a conflict
        resp = await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        assert store.is_paused(conn, "comp1") is True


@pytest.mark.asyncio
async def test_resume_clears_pause(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        assert store.is_paused(conn, "comp1") is True

        resp = await client.post(
            "/v1/competitions/comp1/resume", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        body = await resp.json()
        assert body["paused"] is False
        assert body["paused_at"] is None
        assert store.is_paused(conn, "comp1") is False


@pytest.mark.asyncio
async def test_pause_refuses_finished_competition(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        store.mark_scored(conn, "comp1", status="scored")

        resp = await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 409
        assert store.is_paused(conn, "comp1") is False


@pytest.mark.asyncio
async def test_pause_does_not_block_resume_of_finished_competition(conn, monkeypatch):
    """Resume has no guard — unpausing anything is always safe."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        store.mark_scored(conn, "comp1", status="scored")

        resp = await client.post(
            "/v1/competitions/comp1/resume", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        assert store.is_paused(conn, "comp1") is False


@pytest.mark.asyncio
async def test_reset_scoring_clears_pause(conn, monkeypatch):
    """reset-scoring drops the scored_competitions row, so it implicitly resumes."""
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        assert store.is_paused(conn, "comp1") is True

        resp = await client.post(
            "/v1/competitions/comp1/reset-scoring", headers={"Authorization": "Bearer secret-key"}
        )
        assert resp.status == 200
        assert store.is_paused(conn, "comp1") is False


@pytest.mark.asyncio
async def test_pause_surfaces_in_full_state(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_API_KEY", "secret-key")
    async with TestClient(TestServer(make_app(conn))) as client:
        await _seed_open_competition(client)
        await client.post(
            "/v1/competitions/comp1/pause", headers={"Authorization": "Bearer secret-key"}
        )
        resp = await client.get("/v1/state/competitions/comp1")
        assert resp.status == 200
        assert (await resp.json())["paused_at"] is not None


# ── /admin, /v1/admin/* (wallet-signed create) ───────────────────────

def future_spec(comp_id="new1", **overrides):
    # FakeSubtensor is at block 500, so the schedule must start after it.
    return make_spec_json(comp_id, start_block=600, commit_end_block=700,
                          scoring_end_block=1000, **overrides)


def signed(spec, keypair=ADMIN, receiver=None, path="/v1/admin/competitions"):
    body = json.dumps(spec).encode()
    headers = http_auth.sign(keypair, method="POST", path=path, body=body,
                             receiver_ss58=receiver or LEADER.ss58_address)
    return body, {**headers, "Content-Type": "application/json"}


def browser_signed(spec, keypair=ADMIN):
    """Sign like a wallet extension's signRaw does: over <Bytes>payload</Bytes>."""
    body = json.dumps(spec).encode()
    nonce = time.time_ns()
    payload = http_auth.build_payload(
        scheme="sr25519", method="POST", path="/v1/admin/competitions", body=body,
        nonce_ns=nonce, sender_ss58=keypair.ss58_address, receiver_ss58=LEADER.ss58_address,
    )
    signature = bytes(keypair.sign(b"<Bytes>" + payload + b"</Bytes>"))
    return body, {
        "Content-Type": "application/json",
        http_auth.HEADER_VERSION: "1",
        http_auth.HEADER_HOTKEY: keypair.ss58_address,
        http_auth.HEADER_RECEIVER: LEADER.ss58_address,
        http_auth.HEADER_NONCE: str(nonce),
        http_auth.HEADER_SIGNATURE: "0x" + signature.hex(),
    }


@pytest.fixture
def admin_signers(monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_SIGNERS", {ADMIN.ss58_address})


@pytest.mark.asyncio
async def test_admin_page_hidden_without_signers(conn, monkeypatch):
    monkeypatch.setattr(validator_settings, "ADMIN_SIGNERS", set())
    async with TestClient(TestServer(make_app(conn))) as client:
        assert (await client.get("/admin")).status == 404
        body, headers = signed(future_spec())
        assert (await client.post("/v1/admin/competitions", data=body, headers=headers)).status == 503


@pytest.mark.asyncio
async def test_admin_page_served_with_signers(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.get("/admin")
        assert resp.status == 200
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert "Sign &amp; create" in await resp.text()


@pytest.mark.asyncio
async def test_admin_head(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        body = await (await client.get("/v1/admin/head")).json()
        assert body["block"] == 500 and body["receiver"] == LEADER.ss58_address


@pytest.mark.asyncio
async def test_signed_create_succeeds_then_listed(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec())
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 201, await resp.text()
        assert (await client.get("/v1/competitions/new1")).status == 200


@pytest.mark.asyncio
async def test_browser_wrapped_signature_accepted(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = browser_signed(future_spec())
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 201, await resp.text()


@pytest.mark.asyncio
async def test_signed_create_never_overwrites(conn, admin_signers):
    store.upsert_competition(conn, CompetitionSpec.model_validate(future_spec(name="Original")))
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec(name="Replacement"))
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 409
    assert store.get_competition(conn, "new1")["name"] == "Original"


@pytest.mark.asyncio
async def test_signed_create_rejects_unsigned(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        resp = await client.post("/v1/admin/competitions", json=future_spec())
        assert resp.status == 401


@pytest.mark.asyncio
async def test_signed_create_rejects_unlisted_signer(conn, admin_signers):
    stranger = _keypair(sp_core.Keypair.generate_mnemonic(12))
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec(), keypair=stranger)
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 403
    assert store.get_competition(conn, "new1") is None
    # Rejected before verification, so the stranger's nonce was never stored.
    assert (stranger.ss58_address, int(headers[http_auth.HEADER_NONCE])) not in api_module._signed_nonces._seen


@pytest.mark.asyncio
async def test_signed_create_accepts_signer_under_other_ss58_prefix(conn, monkeypatch):
    other_prefix = sp_core.ss58_encode(sp_core.ss58_decode(ADMIN.ss58_address), 0)
    monkeypatch.setattr(validator_settings, "ADMIN_SIGNERS", {other_prefix})
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec())
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 201, await resp.text()


@pytest.mark.asyncio
async def test_signed_create_rejects_tampered_body(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        _, headers = signed(future_spec())
        tampered = json.dumps(future_spec(emission_weight=1.0)).encode()
        resp = await client.post("/v1/admin/competitions", data=tampered, headers=headers)
        assert resp.status == 401


@pytest.mark.asyncio
async def test_signed_create_rejects_other_receiver(conn, admin_signers):
    other_leader = _keypair(sp_core.Keypair.generate_mnemonic(12))
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec(), receiver=other_leader.ss58_address)
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 401


@pytest.mark.asyncio
async def test_signed_create_rejects_replay(conn, admin_signers):
    async with TestClient(TestServer(make_app(conn))) as client:
        body, headers = signed(future_spec())
        assert (await client.post("/v1/admin/competitions", data=body, headers=headers)).status == 201
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 401


@pytest.mark.asyncio
@pytest.mark.parametrize("overrides", [
    {"start_block": 400},                                    # already started
    {"commit_end_block": 550},                               # commit end before start
    {"scoring_end_block": 800},                              # inside reveal grace
    {"id": "../bad"},
    {"benchmarks": [{"name": "mmlu", "min_score": 0.1}, {"name": "mmlu", "min_score": 0.2}]},
    {"benchmarks": [{"name": "mmlu", "min_score": 2}]},
    {"benchmarks": [{"name": "fluid_knowledge", "min_score": 0}]},
    {"top_n": 2, "emission_distribution": [1.5, -0.5]},
    {"emission_distribution": [float("nan")]},
    {"reveal_grace_blocks": -500},
    {"competition_type": "ram_ceiling", "max_memory_kb": -1},
])
async def test_signed_create_rejects_bad_specs(conn, admin_signers, overrides):
    spec = {**future_spec(), **overrides}
    async with TestClient(TestServer(make_app(conn))) as client:
        body = json.dumps(spec).encode()  # allow_nan: NaN must be rejected server-side
        headers = {**http_auth.sign(ADMIN, method="POST", path="/v1/admin/competitions", body=body,
                                    receiver_ss58=LEADER.ss58_address), "Content-Type": "application/json"}
        resp = await client.post("/v1/admin/competitions", data=body, headers=headers)
        assert resp.status == 422, await resp.text()
    assert store.list_competitions(conn) == []



# ── KL check creation rules ───────────────────────────────────────────────────

def _kl_spec(**overrides) -> CompetitionSpec:
    fields = dict(
        id="kl1", name="kl1", start_block=1000, commit_end_block=2000, scoring_end_block=3000,
        top_n=1, emission_distribution=[1.0],
        benchmarks=[BenchmarkTask(name="mmlu", min_score=0.5)],
        competition_type=CompetitionType.RAM_CEILING, max_memory_kb=4194304,
    )
    fields.update(overrides)
    return CompetitionSpec(**fields)


def test_competition_without_a_kl_corpus_is_unaffected():
    assert creation_errors(_kl_spec(), 500) == []


def test_kl_corpus_with_a_base_pin_is_accepted():
    spec = _kl_spec(kl_corpus_id="corpus-000002", model_repo="Qwen/Qwen2.5-0.5B-Instruct")
    assert creation_errors(spec, 500) == []


def test_kl_corpus_without_a_base_pin_is_refused():
    """Without model_repo the KL run names its own base, so the multiplier is free."""
    errors = creation_errors(_kl_spec(kl_corpus_id="corpus-000002"), 500)
    assert any("model_repo" in e for e in errors), errors


def test_kl_corpus_on_benchmark_floor_is_refused():
    """benchmark_floor ranks on negated memory and never reads benchmark_min."""
    spec = _kl_spec(
        kl_corpus_id="corpus-000002", model_repo="a/b",
        competition_type=CompetitionType.BENCHMARK_FLOOR, max_memory_kb=None,
    )
    errors = creation_errors(spec, 500)
    assert any("benchmark_floor" in e for e in errors), errors
