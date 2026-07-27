# Contributor Guide

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/)
- Docker + Docker Compose
- [`btcli`](https://docs.bittensor.com/btcli) on PATH

## Local blockchain setup

Start a fast-runtime subtensor localnet, create dev wallets, register subnet and participants:

```bash
./scripts/dev.sh
```

This does in order:
1. Starts subtensor via `docker/localnet/docker-compose.yml`
2. Runs `scripts/setup-localnet.sh` — creates wallets (alice/bob/charlie), creates subnet on netuid 2, registers validator (bob) and miner (charlie), starts emissions
3. Starts the validator with `TPN_DOTENV_PATH=.env.localnet --clean`, waits for its health endpoint (port 9100)
4. Seeds the reference competition via `scripts/push_competition.py`, then leaves the validator running

Dev wallets are stored in `./wallets/` (not `~/.bittensor/wallets`).

| Wallet | Role | Dev URI |
|---|---|---|
| `alice` | Subnet owner | `//Alice` |
| `bob` | Validator | `//Bob` |
| `charlie` | Miner | `//Charlie` |

Chain endpoint: `ws://localhost:9946`  
Netuid: `2`

## Validator env for localnet

```bash
# .env.localnet
BITTENSOR=True
NETUID=2
NETWORK=ws://localhost:9946
WALLET_COLDKEY=bob
WALLET_HOTKEY=default
WALLET_PATH=./wallets
VALIDATOR_MODE=leader
ADMIN_API_KEY=localnet-dev-key
```

Start validator manually against localnet:

```bash
TPN_DOTENV_PATH=.env.localnet uv run --package validator python src/validator/main.py --clean
```

Seed it with the reference competition (one-time, or after `--clean`) — this is what
`dev.sh` runs automatically, `push_competition.py` upserts a single spec file by `id`:

```bash
ADMIN_API_KEY=localnet-dev-key uv run scripts/push_competition.py \
  --leader-url http://localhost:9200 \
  competitions/localnet/tpn-localnet.json
```

`scripts/seed_competitions.py` is a separate, one-time bulk-migration script — it reads
an `index.json` and POSTs every listed spec in a directory. Use it only when seeding a
leader from scratch with the full mainnet competition set (`--dir competitions`), not
for the single-file localnet iteration loop.

## CLI against localnet

```bash
# List localnet competitions
uv run --package cli tpn \
  --network ws://localhost:9946 \
  --netuid 2 \
  --leader-url http://localhost:9200 \
  competitions

# Register miner
uv run --package cli tpn \
  --network ws://localhost:9946 \
  --netuid 2 \
  --wallet-path ./wallets \
  register --wallet charlie

# Submit a commit
uv run --package cli tpn \
  --network ws://localhost:9946 \
  --netuid 2 \
  --block-time 0.300 \
  --leader-url http://localhost:9200 \
  --wallet-path ./wallets \
  commit -w charlie -c tpn-localnet
```

## Local competitions

Competition config lives in the leader validator's SQLite store, not in this repo,
served over `GET /v1/competitions` and written via the bearer-token-gated
`POST /v1/competitions` (see `src/validator/README.md`). `competitions/` (including
`competitions/localnet/`) is kept only as historical/seed reference — no running code
reads it anymore.

To change a competition's parameters, re-POST its spec (equivalent to
`push_competition.py`, shown above, or raw curl):

```bash
curl -X POST http://localhost:9200/v1/competitions \
  -H "Authorization: Bearer localnet-dev-key" \
  -H "Content-Type: application/json" \
  -d @competitions/localnet/tpn-localnet.json
```

This upserts by `id`, so editing the JSON file and re-running the command is the
localnet iteration loop — no validator restart needed, but the CLI/follower client
cache (`leader_config_client.py`, 10 min TTL) means changes may take a few minutes
to show up unless you pass `--refresh` (CLI `competitions` command) or restart.

## Project structure

```
competitions/          Competition index + specs (mainnet and localnet)
docker/                Docker configs (localnet subtensor)
scripts/               Operational scripts (autoupdater, dev setup)
shared/common/         Chain, models, settings shared by all packages
shared/competition/    Competition specs, scoring, model store
shared/validation/     Provenance + RAM precheck service (Docker)
src/cli/               Miner CLI (tpn)
src/validator/         Validator
install_cli.sh         Installs uv + the tpn CLI
wallets/               Dev wallets (git-ignored)
```


## Sync dependencies

```bash
uv sync
```

## Testing

See `docs/Testing.md` for the unit test suite and how it relates to this
localnet setup.
