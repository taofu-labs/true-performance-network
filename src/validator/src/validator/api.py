"""
Leader validator's HTTP API.

Three namespaces:
  /v1/follower/*     — minimal, stable contract followers depend on to set weights.
  /v1/state/*        — full detail (reveals, precheck outcomes, weights history,
                       task status) for dashboards. /v1/state/competitions/{id}
                       is a superset of what /v1/follower/scoring-results returns,
                       so dashboards use this instead of stitching both namespaces.
  /v1/competitions/* — competition config. GET is public read. POST (create/update)
                       and DELETE require Authorization: Bearer <ADMIN_API_KEY> —
                       these are the only privileged, write-capable routes the
                       leader exposes.

/v1/follower/*, /v1/state/*, and GET /v1/competitions/* have no auth — intended to
sit behind a proxy and be treated as public read access. POST and DELETE on
/v1/competitions MUST be served over HTTPS in any real deployment — a bearer token
sent over plain HTTP is sniffable in transit, defeating the point of the check.
"""
import hmac

from aiohttp import web
from loguru import logger

from common.chain import current_block as get_current_block
from common.models.competition import CompetitionSpec
from validator import store
from validator import settings as validator_settings


MAX_REQUEST_BODY_BYTES = 256 * 1024


def _require_admin_auth(handler):
    """Gate a handler behind Authorization: Bearer <ADMIN_API_KEY>.

    Fails closed: an unset ADMIN_API_KEY returns 503 (endpoint not configured)
    rather than silently accepting all requests. Constant-time compare on the
    token to avoid a timing side-channel on the secret.
    """
    async def wrapped(request: web.Request) -> web.Response:
        expected = validator_settings.ADMIN_API_KEY
        if not expected:
            return web.json_response({"error": "admin API not configured"}, status=503)
        given = request.headers.get("Authorization", "")
        token = given[len("Bearer "):] if given.startswith("Bearer ") else ""
        if not token or not hmac.compare_digest(token, expected):
            return web.json_response({"error": "unauthorized"}, status=401)
        return await handler(request)
    return wrapped


class LeaderApiMixin:
    leader_api_runner: web.AppRunner | None = None
    leader_api_site: web.TCPSite | None = None

    async def _start_leader_api(self):
        app = web.Application(client_max_size=MAX_REQUEST_BODY_BYTES)
        app.router.add_get("/v1/follower/scoring-results", self._h_follower_scoring_results)
        app.router.add_get("/v1/state/competitions/{competition_id}", self._h_state_competition)
        app.router.add_get("/v1/state/status", self._h_state_status)
        app.router.add_get("/v1/state/weights-history", self._h_state_weights_history)
        app.router.add_get("/v1/competitions", self._h_list_competitions)
        app.router.add_get("/v1/competitions/{competition_id}", self._h_get_competition)
        app.router.add_post("/v1/competitions", _require_admin_auth(self._h_upsert_competition))
        app.router.add_delete(
            "/v1/competitions/{competition_id}", _require_admin_auth(self._h_delete_competition)
        )
        app.router.add_post(
            "/v1/competitions/{competition_id}/reset-scoring",
            _require_admin_auth(self._h_reset_competition_scoring),
        )

        self.leader_api_runner = web.AppRunner(app)
        await self.leader_api_runner.setup()

        self.leader_api_site = web.TCPSite(
            self.leader_api_runner, validator_settings.LEADER_API_HOST, validator_settings.LEADER_API_PORT
        )
        await self.leader_api_site.start()
        logger.info(
            f"Leader API started on http://{validator_settings.LEADER_API_HOST}:{validator_settings.LEADER_API_PORT}"
        )

    # ── /v1/follower/* ────────────────────────────────────────────────

    async def _h_follower_scoring_results(self, request: web.Request) -> web.Response:
        competition_id = request.query.get("competition_id")
        if not competition_id:
            return web.json_response({"error": "competition_id is required"}, status=400)
        results = (
            store.scoring_results_for_competition(self._db, competition_id)
            if store.get_stage(self._db, competition_id) == "finalized"
            else []
        )
        return web.json_response({
            "results": results,
            "scored_status": store.scored_status(self._db, competition_id),
        })

    # ── /v1/state/* ────────────────────────────────────────────────────

    async def _h_state_competition(self, request: web.Request) -> web.Response:
        competition_id = request.match_info["competition_id"]
        if store.get_competition(self._db, competition_id) is None:
            return web.json_response({"error": "competition not found"}, status=404)
        return web.json_response(store.full_state_for_competition(self._db, competition_id))

    async def _h_state_status(self, request: web.Request) -> web.Response:
        status = await self.get_validator_status()
        return web.json_response(status)

    async def _h_state_weights_history(self, request: web.Request) -> web.Response:
        competition_id = request.query.get("competition_id")
        if not competition_id:
            return web.json_response({"error": "competition_id is required"}, status=400)
        weights_history = store.weights_history_for_competition(self._db, competition_id)
        return web.json_response({"weights_history": weights_history})

    # ── /v1/competitions/* ───────────────────────────────────────────────

    async def _h_list_competitions(self, request: web.Request) -> web.Response:
        specs = store.list_competitions(self._db)
        if request.query.get("active") == "true":
            block_param = request.query.get("block")
            if block_param is not None:
                try:
                    current_block = int(block_param)
                except ValueError:
                    return web.json_response(
                        {"error": f"block must be an integer, got {block_param!r}"}, status=400
                    )
                if current_block < 0:
                    return web.json_response(
                        {"error": f"block must be non-negative, got {current_block}"}, status=400
                    )
            else:
                current_block = get_current_block(self.subtensor)
            specs = [s for s in specs if CompetitionSpec.model_validate(s).is_active(current_block)]
        return web.json_response({"competitions": specs})

    async def _h_get_competition(self, request: web.Request) -> web.Response:
        competition_id = request.match_info["competition_id"]
        spec = store.get_competition(self._db, competition_id)
        if spec is None:
            return web.json_response({"error": "competition not found"}, status=404)
        return web.json_response(spec)

    async def _h_upsert_competition(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except web.HTTPRequestEntityTooLarge:
            return web.json_response(
                {"error": f"request body exceeds {MAX_REQUEST_BODY_BYTES} bytes"}, status=413
            )
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        try:
            spec = CompetitionSpec.model_validate(data)
        except Exception as e:
            return web.json_response({"error": f"invalid competition spec: {e}"}, status=422)
        store.upsert_competition(self._db, spec)
        return web.json_response(spec.model_dump(mode="json"))

    async def _h_delete_competition(self, request: web.Request) -> web.Response:
        """Remove a competition spec. 409 if it has scoring state, unless ?force=true."""
        competition_id = request.match_info["competition_id"]
        if store.get_competition(self._db, competition_id) is None:
            return web.json_response({"error": "competition not found"}, status=404)

        force = request.query.get("force") == "true"
        if not force and store.competition_has_state(self._db, competition_id):
            return web.json_response(
                {
                    "error": "competition has scoring state; refusing to delete",
                    "hint": "retry with ?force=true to delete the spec anyway "
                            "(scoring rows for this id are left in place)",
                },
                status=409,
            )

        store.delete_competition(self._db, competition_id)
        logger.warning(f"Competition {competition_id} deleted via admin API (force={force})")
        return web.json_response({"deleted": competition_id, "forced": force})

    RESETTABLE_STAGES = ("stage1_ranking", "failed_no_reveals",
                         "failed_no_participants", "failed_stage1_infra")

    async def _h_reset_competition_scoring(self, request: web.Request) -> web.Response:
        """Clear scoring state so a failed competition re-runs from stage 1.

        Only a competition that failed, or is still in stage 1, can be reset —
        and only while its scoring window is open. There is no override.
        """
        competition_id = request.match_info["competition_id"]
        spec_row = store.get_competition(self._db, competition_id)
        if spec_row is None:
            return web.json_response({"error": "competition not found"}, status=404)

        stage = store.get_stage(self._db, competition_id)
        if stage not in self.RESETTABLE_STAGES:
            return web.json_response(
                {
                    "error": f"competition is in stage {stage!r}; only a failed or "
                             f"stage-1 competition can be reset",
                    "stage": stage,
                    "scored_status": store.scored_status(self._db, competition_id),
                },
                status=409,
            )

        spec = CompetitionSpec.model_validate(spec_row)
        try:
            current_block = get_current_block(self.subtensor)
        except Exception as e:
            return web.json_response(
                {"error": f"cannot read current block to check scoring window: {e}"},
                status=503,
            )
        if current_block >= spec.scoring_end_block:
            return web.json_response(
                {
                    "error": "scoring window has closed; a reset cannot produce new results",
                    "current_block": current_block,
                    "scoring_end_block": spec.scoring_end_block,
                },
                status=409,
            )

        previous_status = store.scored_status(self._db, competition_id)
        deleted = store.reset_competition_scoring(self._db, competition_id)
        logger.warning(
            f"Competition {competition_id} scoring state reset via admin API "
            f"(previous stage={stage}, status={previous_status}, deleted={deleted})"
        )
        return web.json_response({
            "reset": competition_id,
            "previous_stage": stage,
            "previous_status": previous_status,
            "deleted": deleted,
        })
