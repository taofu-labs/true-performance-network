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
| `WALLET_COLDKEY` | `my-validator` | Coldkey name in `~/.bittensor/wallets` |
| `WALLET_HOTKEY` | `default` | Hotkey name |
| `WALLET_PATH` | (bittensor default) | Override wallet directory |
| `COMPETITION_INDEX_URL` | GitHub raw URL | Points to `competitions/index.json` |
| `LAUNCH_HEALTH` | `True` | Enable health endpoint on port 9100 |

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

**Direct (no autoupdate):**

```bash
./start_validator.sh
```

Passes all extra args to the validator:
```bash
./start_validator.sh --clean
```

**Custom `.env` path:**

```bash
TPN_DOTENV_PATH=/etc/tpn/prod.env ./start_validator.sh
TPN_DOTENV_PATH=/etc/tpn/prod.env ./scripts/start_autoupdater_pm2.sh
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

This includes ban records and scored submission history, used to avoid rescoring and enforce bans across restarts. Wipe with `--clean` flag on startup.

## pm2 management

```bash
pm2 status
pm2 logs tpn_validator
pm2 stop tpn_validator
pm2 restart tpn_validator
pm2 delete tpn_validator
```
