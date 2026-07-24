# TPN Validator

Two validator modes, controlled by `VALIDATOR_MODE` (`.env`):

- **leader** (default) — scans chain reveals, prechecks and benchmarks submissions,
  computes and sets weights, persists every scoring run to SQLite, and serves a
  read-only HTTP API for followers and dashboards.
- **follower** — parses competitions itself but reads scoring results from a leader
  validator's API instead of scanning/prechecking/benchmarking. Recomputes emission
  weights locally from the leader's `ScoringResult`s and sets its own weights.

Both modes run the same chain-copy `weight_loop()` fallback in parallel — if a
follower can't reach its leader, it just skips that competition's cycle and keeps
submitting stake-weighted consensus weights from chain until the leader is reachable
again.

## Persistence

All validator state (scored competitions, bans, and — leader-only — full scoring
history and weights history) lives in a single SQLite file:

```
~/.tpn/validator-storage/validator.db      # POSIX
%APPDATA%/tpn/validator-storage/validator.db  # Windows
```

`python main.py --clean` wipes it before starting.

## Configuration

### Common (both modes)

```
VALIDATOR_MODE=leader              # "leader" | "follower"
```

### Leader mode

```
LEADER_API_HOST=0.0.0.0
LEADER_API_PORT=9200
```

```
ADMIN_API_KEY=                     # required to enable POST /v1/competitions
```

Run the leader's API behind a reverse proxy (HTTPS). The read endpoints
(`/v1/follower/*`, `/v1/state/*`, `GET /v1/competitions*`) have no auth — treat them
as public read access. `POST /v1/competitions` is bearer-token gated
(`Authorization: Bearer <ADMIN_API_KEY>`) and **must** be served over HTTPS in any
real deployment — a bearer token sent over plain HTTP is sniffable in transit.
Generate the key once with `python -c "import secrets; print(secrets.token_urlsafe(32))"`
and store it as a deploy secret. Leaving `ADMIN_API_KEY` unset disables the write
endpoint (503), it does not make it public.

### Follower mode

```
LEADER_VALIDATOR_URL=https://leader.tpn.internal   # required
FOLLOWER_POLL_INTERVAL=60
```

## Leader API

`GET /v1/follower/scoring-results?competition_id=X&since_run_id=0`
Runs for a competition newer than `since_run_id`, each with embedded `ScoringResult`s.
This is what followers poll to set weights.

`GET /v1/state/competitions/{competition_id}`
Full detail for a competition: every scoring run's reveals, precheck outcomes
(including SKIPPED/DISQUALIFIED reasons), scoring results, and weights history.
Superset of the follower endpoints — dashboards should use this instead of
stitching `/v1/follower/*` calls together.

`GET /v1/state/status`
Leader's task/loop status (same shape as `get_validator_status()`).

`GET /v1/state/weights-history?competition_id=X`
Past weight submissions for a competition.

`GET /v1/competitions`
All competitions. `?active=true&block=N` filters to active ones at block `N`
(current chain block if `block` omitted). This is what followers, the miner CLI,
and dashboards should read competition config from.

`GET /v1/competitions/{competition_id}`
Single competition spec. 404 if not found.

`POST /v1/competitions` (requires `Authorization: Bearer <ADMIN_API_KEY>`)
Create or update (upsert by `id`) one competition. Body is a full `CompetitionSpec`
JSON object — validated the same way as `competitions/*.json` was before this moved
off GitHub. 401 on missing/invalid token, 503 if `ADMIN_API_KEY` isn't set, 422 on a
spec that fails validation.
