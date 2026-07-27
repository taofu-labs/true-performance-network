#!/usr/bin/env bash
set -euo pipefail

echo "==> Starting subtensor localnet..."
docker compose -f docker/localnet/docker-compose.yml up -d

echo "==> Setting up chain (subnet + wallets)..."
./scripts/setup-localnet.sh

echo "==> Starting TPN validator..."
TPN_DOTENV_PATH=.env.localnet uv run --package validator python src/validator/main.py --clean &
VALIDATOR_PID=$!

cleanup() {
    echo "==> Stopping validator (pid $VALIDATOR_PID)..."
    kill "$VALIDATOR_PID" 2>/dev/null || true
    wait "$VALIDATOR_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "==> Waiting for validator health endpoint..."
for _ in $(seq 1 60); do
    if curl -sf http://localhost:9100/health >/dev/null 2>&1; then
        break
    fi
    sleep 1
done
if ! curl -sf http://localhost:9100/health >/dev/null 2>&1; then
    echo "ERROR: validator did not become healthy within 60s" >&2
    exit 1
fi

echo "==> Seed Competitions..."
ADMIN_API_KEY=localnet-dev-key uv run scripts/push_competition.py --leader-url http://localhost:9200 competitions/localnet/tpn-localnet.json

echo "==> Validator running (pid $VALIDATOR_PID). Ctrl+C to stop."
wait "$VALIDATOR_PID"
