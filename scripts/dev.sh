#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting subtensor localnet..."
docker compose -f docker/localnet/docker-compose.yml up -d

echo "==> Setting up chain (subnet + wallets)..."
./scripts/setup-localnet.sh

echo "==> Starting TPN validator..."
TPN_DOTENV_PATH=.env.localnet uv run --package validator python src/validator/main.py --clean
