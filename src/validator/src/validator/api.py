"""
Leader validator's HTTP API.

Three namespaces:
  /v1/follower/*     — minimal, stable contract followers depend on to set weights.
  /v1/state/*        — full detail (reveals, precheck outcomes, weights history,
                       task status) for dashboards. /v1/state/competitions/{id}
                       is a superset of what /v1/follower/scoring-results returns,
                       so dashboards use this instead of stitching both namespaces.
  /v1/competitions/* — competition config. GET is public read. POST (create/update)
                       requires Authorization: Bearer <ADMIN_API_KEY> — this is the
                       only privileged, write-capable route the leader exposes.

/v1/follower/*, /v1/state/*, and GET /v1/competitions/* have no auth — intended to
sit behind a proxy and be treated as public read access. POST /v1/competitions MUST
be served over HTTPS in any real deployment — a bearer token sent over plain HTTP is
sniffable in transit, defeating the point of the check.
"""
import hmac

from aiohttp import web
from loguru import logger

from common.models.competition import CompetitionSpec
from validator import store
from validator import settings as validator_settings


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
        app = web.Application()
        app.router.add_get("/v1/follower/scoring-results", self._h_follower_scoring_results)
        app.router.add_get("/v1/state/competitions/{competition_id}", self._h_state_competition)
        app.router.add_get("/v1/state/status", self._h_state_status)
        app.router.add_get("/v1/state/weights-history", self._h_state_weights_history)
        app.router.add_get("/v1/competitions", self._h_list_competitions)
        app.router.add_get("/v1/competitions/{competition_id}", self._h_get_competition)
        app.router.add_post("/v1/competitions", _require_admin_auth(self._h_upsert_competition))

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
        since_run_id = int(request.query.get("since_run_id", 0))
        runs = store.scoring_results_since(self._db, competition_id, since_run_id)
        return web.json_response({
            "runs": runs,
            "scored_status": store.scored_status(self._db, competition_id),
        })

    # ── /v1/state/* ────────────────────────────────────────────────────

    async def _h_state_competition(self, request: web.Request) -> web.Response:
        competition_id = request.match_info["competition_id"]
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
            current_block = int(block_param) if block_param else self.subtensor.block()
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
        except Exception:
            return web.json_response({"error": "invalid JSON body"}, status=400)
        try:
            spec = CompetitionSpec.model_validate(data)
        except Exception as e:
            return web.json_response({"error": f"invalid competition spec: {e}"}, status=422)
        store.upsert_competition(self._db, spec)
        return web.json_response(spec.model_dump(mode="json"))
