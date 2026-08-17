# Running a Validator

## Prerequisites

- Linux host with sudo access
- `git`, `curl`, build tools, Python 3.12, Node/npm, `uv`, PM2, and `btcli`
- Bittensor wallet present on the host, normally under `~/.bittensor/wallets`
- Validator hotkey registered on the TPN subnet (see `btcli subnets register`)
- Only the primary scoring validator should use `VALIDATOR_MODE=leader`
- Every additional validator should use `VALIDATOR_MODE=follower`

## Follower hardware

Follower validators do not download models, run Docker prechecks, or benchmark
submissions. They read scoring results from the primary validator and submit
weights on chain.

Minimum:

- 2 vCPU
- 4 GB RAM
- 20-40 GB disk
- Stable outbound network

Comfortable:

- 4 vCPU
- 8 GB RAM
- 50+ GB disk

No GPU is required. Docker is not required for follower mode.

## Install dependencies

Ubuntu/Debian follower example:

```bash
sudo apt-get update
sudo apt-get install -y \
  git curl ca-certificates build-essential pkg-config libssl-dev \
  python3 python3-dev python3-venv python3-pip \
  nodejs npm

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv python install 3.12
sudo npm install -g pm2
sudo env PATH="$PATH:/usr/bin:/usr/local/bin:/bin" pm2 startup systemd -u "$USER" --hp "$HOME"
```

Install `btcli` if it is not already on `PATH`:

```bash
mkdir -p "$HOME/.local/bin"
python3 -m venv "$HOME/.local/share/truepn/bittensor-venv"
"$HOME/.local/share/truepn/bittensor-venv/bin/python" -m pip install --upgrade pip wheel setuptools
"$HOME/.local/share/truepn/bittensor-venv/bin/python" -m pip install --upgrade bittensor bittensor-cli
ln -sf "$HOME/.local/share/truepn/bittensor-venv/bin/btcli" "$HOME/.local/bin/btcli"
```

## Prepare repo

Clone the repo, then sync the workspace:

```bash
git clone https://github.com/taofu-labs/tao-performance-network.git
cd tao-performance-network
uv sync
```

Create the validator env file:

```bash
cp .env.example .env
```

For a normal follower validator on mainnet, set:

```dotenv
VALIDATOR_MODE=follower
LEADER_VALIDATOR_URL=https://val0.trueperformancenetwork.com
WALLET_COLDKEY=<coldkey>
WALLET_HOTKEY=<hotkey>
```

For a testnet follower, use:

```dotenv
VALIDATOR_MODE=follower
LEADER_VALIDATOR_URL=https://tval0.trueperformancenetwork.com
NETWORK=test
NETUID=533
WALLET_COLDKEY=<coldkey>
WALLET_HOTKEY=<hotkey>
```

If your wallet directory is not the Bittensor default, also set:

```dotenv
WALLET_PATH=/path/to/wallets
```

Do not set `VALIDATOR_MODE=leader` unless you are operating the primary scoring
validator. Running multiple leaders can produce conflicting scoring state.

## Optional env vars

| Variable | Default | Description |
|---|---|---|
| `NETWORK` | `finney` | Chain endpoint or `ws://...` |
| `NETUID` | `65` | Subnet UID |
| `WALLET_PATH` | (bittensor default) | Override wallet directory |
| `LAUNCH_HEALTH` | `False` | Set `True` to enable health endpoint on port 9100 |
| `FOLLOWER_POLL_INTERVAL` | `60` | Seconds between polls of the leader's scoring results |

Competition configs and scoring results come from the primary validator API.
Followers read them over `GET /v1/competitions` and follower scoring endpoints.
See `src/validator/README.md` for API details and primary-validator settings.

## Running

**Production (pm2, with autoupdate):**

```bash
./scripts/start_autoupdater_pm2.sh
pm2 save
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

This includes local scored submission history. On the primary scoring validator,
it also includes full scoring runs, ban records, and weights history. Wipe with
`--clean` flag on startup.

## pm2 management

```bash
pm2 status
pm2 logs tpn_validator
pm2 stop tpn_validator
pm2 restart tpn_validator
pm2 delete tpn_validator
```