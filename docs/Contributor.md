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
3. Starts the validator with `TPN_DOTENV_PATH=.env.localnet`

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
COMPETITION_INDEX_URL=./competitions/localnet/index.json
```

Start validator manually against localnet:

```bash
TPN_DOTENV_PATH=.env.localnet uv run --package validator python src/validator/main.py --clean
```

## CLI against localnet

```bash
# List localnet competitions
uv run --package cli tpn \
  --network ws://localhost:9946 \
  --netuid 2 \
  --competition-url ./competitions/localnet/index.json \
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
  --competition-url ./competitions/localnet/index.json \
  --wallet-path ./wallets \
  commit -w charlie -c tpn-localnet
```

## Local competitions

Localnet competition config lives in `competitions/localnet/`. The index at `competitions/localnet/index.json` lists `tpn-localnet.json`.

Edit `competitions/localnet/tpn-localnet.json` to iterate on competition parameters — the validator and CLI reload on each cycle when using a local `--competition-url` path, no restart needed.

Mainnet configs live in `competitions/` and are fetched from GitHub raw URLs by default.

## Project structure

```
competitions/          Competition index + specs (mainnet and localnet)
docker/                Docker configs (localnet subtensor)
scripts/               Operational scripts (autoupdater, dev setup)
shared/common/         Chain, models, settings shared by all packages
shared/competition/    Competition specs, scoring, model store
src/cli/               Miner CLI (tpn)
src/validator/         Validator
wallets/               Dev wallets (git-ignored)
```


## Sync dependencies

```bash
uv sync
```
