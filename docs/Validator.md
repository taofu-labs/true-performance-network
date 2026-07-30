# Running a Validator

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) installed
- Bittensor wallet registered on the TPN subnet
- Registered hotkey on netuid (see `btcli subnet register`)

## Setup

```bash
cp .env.example .env
# Edit .env — set WALLET_COLDKEY, WALLET_HOTKEY, NETUID
```

Key env vars:

| Variable | Default | Description |
|---|---|---|
| `BITTENSOR` | `True` | Set `False` to run without chain |
| `NETUID` | `0` | Subnet UID |
| `NETWORK` | `finney` | Chain endpoint or `ws://...` |
| `WALLET_COLDKEY` | `test` | Coldkey name in `~/.bittensor/wallets` |
| `WALLET_HOTKEY` | `m1` | Hotkey name |
| `WALLET_PATH` | (bittensor default) | Override wallet directory |
| `LAUNCH_HEALTH` | `False` | Set `True` to enable health endpoint on port 9100 |
| `VALIDATOR_MODE` | `leader` | `leader` or `follower` |

**Leader mode:**

| Variable | Default | Description |
|---|---|---|
| `LEADER_API_HOST` | `0.0.0.0` | Bind host for the leader read/admin API |
| `LEADER_API_PORT` | `9200` | Bind port for the leader read/admin API |
| `ADMIN_API_KEY` | (none) | Bearer token gating `POST /v1/competitions`. Unset makes the write endpoint refuse all requests (503) rather than being open |

**Follower mode:**

| Variable | Default | Description |
|---|---|---|
| `LEADER_VALIDATOR_URL` | (none) | Leader's API base URL to follow |
| `FOLLOWER_POLL_INTERVAL` | `60` | Seconds between polls of the leader's scoring results |

**Scoring / precheck tuning:**

| Variable | Default | Description |
|---|---|---|
| `RAM_CHECK_LYING_TOLERANCE` | `0.01` | Allowed relative diff between self-reported and measured `max_memory` before disqualification |
| `BENCHMARK_BACKEND` | `mock` | `mock` (in-process fake) or `http` (real coordinator) |
| `COORDINATOR_BASE_URL` | (taofulabs bench endpoint) | Benchmark coordinator base URL, `http` backend only |
| `COORDINATOR_API_KEY` | (none) | Auth for the benchmark coordinator, `http` backend only |
| `PRECHECK_IMAGE` | `ghcr.io/taofu-labs/tpn-precheck` | Docker image used for provenance/RAM precheck |
| `PRECHECK_HOST_PORT` | `8081` | Host port the precheck container binds to (loopback-only) |
| `BENCHMARK_POLL_TIMEOUT_SECONDS` | `5400` | Max wall-clock time to poll one benchmark run before giving up |

**Loop timing:**

| Variable | Default | Description |
|---|---|---|
| `VALIDATOR_LOOP_INTERVAL` | `60` (`10` if `BITTENSOR=False`) | Seconds between leader scoring-loop iterations |
| `WEIGHT_SUBMIT_INTERVAL` | `1260` (`10` if `BITTENSOR=False`) | Seconds between on-chain weight submissions |

Competition configs live in the leader's SQLite store now, not GitHub. Leader mode
reads/writes them directly; follower mode and the CLI read them over the leader's
`GET /v1/competitions` API. See `src/validator/README.md` for the full API and
`scripts/seed_competitions.py` for seeding a new leader from the JSON files still
kept in `competitions/` for reference.

## Running

**Production (pm2, with autoupdate):**

```bash
./scripts/start_autoupdater_pm2.sh
```

Starts the validator as a pm2 daemon. Checks for updates every 15 minutes, restarts automatically on new releases.

Pass extra args after `--`:
```bash
./scripts/start_autoupdater_pm2.sh -- --no_autoupdate
./scripts/start_autoupdater_pm2.sh -- --clean
```

## Health endpoint

When `LAUNCH_HEALTH=True`, the validator exposes:

```
GET http://0.0.0.0:9100/health
```

Returns hotkey, registration status, and timestamp. Port configurable via `VALIDATOR_HEALTH_PORT`.

## Local state

The validator stores persistent state in:

- Linux/macOS: `~/.tpn/validator-storage/`
- Windows: `%APPDATA%/tpn/validator-storage/`

This includes ban records, scored submission history, and — in leader mode — full scoring run history and weights history, used to avoid rescoring, enforce bans across restarts, and serve the leader's read API. Wipe with `--clean` flag on startup.

## pm2 management

```bash
pm2 status
pm2 logs tpn_validator
pm2 stop tpn_validator
pm2 restart tpn_validator
pm2 delete tpn_validator
```
