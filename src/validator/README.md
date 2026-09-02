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
LEADER_VALIDATOR_URL=https://val0.trueperformancenetwork.com   # required
FOLLOWER_POLL_INTERVAL=60
```

## Leader API

`GET /v1/follower/scoring-results?competition_id=X`
Final scoring results for one competition. This is what followers poll to set
weights. `competition_id` is required — 400 without it.

```json
{
  "results": [
    {"competition_id": "tpn-001", "hotkey": "5F...", "final_score": -812345,
     "max_memory_kb": 812345, "finalized_at": 1756100000.0}
  ],
  "scored_status": "scored"
}
```

`results` is empty until the competition reaches stage `finalized`, so followers
never see partial rankings. `scored_status` is `null` before scoring starts, then
one of `scored`, `failed_no_reveals`, `failed_no_participants`, or
`failed_stage1_infra`. An empty `results` with a `failed_*` status means the leader
gave up and no weights will ever be published for that competition.

`GET /v1/state/competitions/{competition_id}`
Full detail for a competition: revealed candidates with their status and failure
reasons, benchmark results, scoring results, and weights history. Superset of the
follower endpoints — dashboards should use this instead of stitching
`/v1/follower/*` calls together. 404 if no competition with that id exists.

`GET /v1/state/status`
Leader's task/loop status (same shape as `get_validator_status()`).

`GET /v1/state/weights-history?competition_id=X`
Past weight submissions for a competition.

`GET /v1/competitions`
All competitions. `?active=true&block=N` filters to active ones at block `N`
(current chain block if `block` omitted). This is what followers, the miner CLI,
and dashboards should read competition config from. 400 if `block` is not a
non-negative integer.

`GET /v1/competitions/{competition_id}`
Single competition spec. 404 if not found.

`POST /v1/competitions` (requires `Authorization: Bearer <ADMIN_API_KEY>`)
Create or update (upsert by `id`) one competition. Body is a full `CompetitionSpec`
JSON object — validated the same way as `competitions/*.json` was before this moved
off GitHub. 401 on missing/invalid token, 503 if `ADMIN_API_KEY` isn't set, 422 on a
spec that fails validation. 413 on a body over 256 KB.

`DELETE /v1/competitions/{competition_id}` (requires `Authorization: Bearer <ADMIN_API_KEY>`)
Remove one competition spec — intended for an id pushed by mistake, before anything
has been scored against it. 404 if not found, 401/503 on auth as above.

Refuses with 409 once the competition has scoring state (reveals, benchmark results,
scoring results, weights, or a scoring stage), because those tables key off
`competition_id` with no foreign key — dropping the spec would leave their rows
unreachable but still present. Append `?force=true` to delete the spec anyway;
the scoring rows for that id are left in place.

`POST /v1/competitions/{competition_id}/reset-scoring` (requires `Authorization: Bearer <ADMIN_API_KEY>`)
Clear scoring state so the competition re-runs from stage 1. For a competition that
gave up on a transient failure — `failed_stage1_infra` (coordinator unreachable, chain
scan failed), `failed_no_reveals`, or `failed_no_participants` — while its scoring
window is still open. Without this the only recovery was `--clean`, which wipes every
competition's history plus the ban list.

Deletes that competition's `revealed_candidates`, `benchmark_results`,
`benchmark_runs`, `scoring_results`, and its `scored_competitions` row, and reports
the per-table counts. **Bans and weights history are preserved** — a ban records
proven miner misbehaviour that a validator-side infra failure does not undo, and
weights history stays as an audit trail of what was already set on chain.

**There is no override.** A reset is allowed only when both hold:

- The competition's stage is `stage1_ranking` (including a competition that has not
  been touched yet), `failed_no_reveals`, `failed_no_participants`, or
  `failed_stage1_infra`. A competition in `stage2_scoring` or `finalized` is refused
  with 409 — stage 2 is mid-run, and a finalized competition already had weights
  computed and published from its result.
- Its scoring window is still open (`current_block < scoring_end_block`). Refused with
  409 once closed, since no further reveals can arrive and a re-run cannot produce new
  results.

404 if not found, 503 if the current block can't be read to check the window.

After a reset the leader picks the competition up on its next loop tick; no restart
needed.
