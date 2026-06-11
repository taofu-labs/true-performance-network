#!/usr/bin/env bash
set -euo pipefail

CHAIN="ws://127.0.0.1:9946"
NETUID=2

# Dev accounts — pre-funded at genesis, created via --uri (no mnemonic/transfer needed)
OWNER="alice"      # subnet owner (Alice coldkey = //Alice, pre-funded)
VALIDATOR="bob"    # validator
MINER="charlie"    # miner

# Repo-local wallet dir — keeps localnet keys out of ~/.bittensor
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLET_PATH="$SCRIPT_DIR/../wallets"
mkdir -p "$WALLET_PATH"

# ── Wallet creation ────────────────────────────────────────────────────────────

echo "==> Creating dev wallets in $WALLET_PATH ..."

dev_wallet() {
  local name="$1" uri="$2"
  if [[ -f "$WALLET_PATH/$name/coldkeypub.txt" ]]; then
    echo "    '$name' already exists, skipping"
  else
    # Coldkey from dev URI (pre-funded at genesis); hotkey is a fresh random key
    btcli wallet new_coldkey \
      --wallet.name "$name" \
      --wallet.path "$WALLET_PATH" \
      --uri "$uri" \
      --no-use-password
    btcli wallet new_hotkey \
      --wallet.name "$name" \
      --wallet.path "$WALLET_PATH" \
      --wallet.hotkey default \
      --no-use-password \
      --n-words 12 \
      --quiet
  fi
}

dev_wallet "$OWNER"     "Alice"
dev_wallet "$VALIDATOR" "Bob"
dev_wallet "$MINER"     "Charlie"

# ── Wait for chain ─────────────────────────────────────────────────────────────

HTTP_CHAIN="${CHAIN/ws:\/\//http://}"

echo "==> Waiting for chain to produce blocks..."
until block=$(curl -sf -X POST "$HTTP_CHAIN" \
    -H "Content-Type: application/json" \
    -d '{"jsonrpc":"2.0","id":1,"method":"chain_getBlock","params":[]}' \
    2>/dev/null | python3 -c "
import sys, json
d = json.load(sys.stdin)
n = d['result']['block']['header']['number']
print(int(n, 16))
" 2>/dev/null) && [[ "$block" -ge 2 ]]; do
  sleep 2
done
echo "==> Chain ready (block $block)"

# ── Create subnet (Alice as owner) ────────────────────────────────────────────

if btcli subnet list --subtensor.chain_endpoint "$CHAIN" 2>&1 | grep -q "tpn"; then
  echo "==> TPN subnet already exists, skipping creation"
else
  echo "==> Creating TPN subnet (Alice as owner)..."
  btcli subnet create \
    --wallet.name "$OWNER" \
    --wallet.path "$WALLET_PATH" \
    --wallet.hotkey default \
    --subtensor.chain_endpoint "$CHAIN" \
    --subnet-name "TPN Localnet" \
    --github-repo "https://github.com/taofu-labs/tao-performance-network" \
    --subnet-contact "dev@example.xyz" \
    --subnet-url "https://github.com/taofu-labs/tao-performance-network" \
    --discord-handle "taofu" \
    --description "TPN localnet dev subnet" \
    --logo-url "https://github.com/taofu-labs/tao-performance-network" \
    --additional-info "localnet" \
    --no-mev-protection \
    --no-prompt
fi

# ── Helper: check if hotkey is registered on NETUID ───────────────────────────

is_registered() {
  local wallet="$1"
  btcli wallet overview \
    --wallet.name "$wallet" \
    --wallet.path "$WALLET_PATH" \
    --subtensor.chain_endpoint "$CHAIN" 2>&1 | grep -q "netuid: $NETUID\|uid:"
}

# ── Register validator ─────────────────────────────────────────────────────────

if is_registered "$VALIDATOR"; then
  echo "==> Validator (Bob) already registered on netuid $NETUID, skipping"
else
  echo "==> Registering validator (Bob) on netuid $NETUID..."
  btcli subnet register \
    --wallet.name "$VALIDATOR" \
    --wallet.path "$WALLET_PATH" \
    --wallet.hotkey default \
    --netuid "$NETUID" \
    --subtensor.chain_endpoint "$CHAIN" \
    --no-prompt
fi

# ── Register miner ─────────────────────────────────────────────────────────────

if is_registered "$MINER"; then
  echo "==> Miner (Charlie) already registered on netuid $NETUID, skipping"
else
  echo "==> Registering miner (Charlie) on netuid $NETUID..."
  btcli subnet register \
    --wallet.name "$MINER" \
    --wallet.path "$WALLET_PATH" \
    --wallet.hotkey default \
    --netuid "$NETUID" \
    --subtensor.chain_endpoint "$CHAIN" \
    --no-prompt \
    --quiet
fi

# ── Start subnet emissions (enables staking) ──────────────────────────────────

echo "==> Checking if subnet $NETUID can start..."
btcli subnet check-start \
  --netuid "$NETUID" \
  --network "$CHAIN"

echo "==> Starting subnet $NETUID emissions (Alice as owner)..."
btcli subnet start \
  --wallet.name "$OWNER" \
  --wallet.path "$WALLET_PATH" \
  --wallet.hotkey default \
  --netuid "$NETUID" \
  --subtensor.chain_endpoint "$CHAIN" \
  --no-prompt

# ── Stake validator ────────────────────────────────────────────────────────────

echo "==> Adding stake to validator (Bob) on netuid $NETUID..."
btcli stake add \
  --wallet.name "$VALIDATOR" \
  --wallet.path "$WALLET_PATH" \
  --wallet.hotkey default \
  --subtensor.chain_endpoint "$CHAIN" \
  --netuid "$NETUID" \
  --amount 10 \
  --unsafe \
  --no-mev-protection \
  --no-prompt

# ── Verify ─────────────────────────────────────────────────────────────────────

echo "==> Verifying setup..."
btcli subnet list --subtensor.chain_endpoint "$CHAIN"
btcli wallet overview --wallet.name "$VALIDATOR" --wallet.path "$WALLET_PATH" --subtensor.chain_endpoint "$CHAIN"
btcli wallet overview --wallet.name "$MINER"     --wallet.path "$WALLET_PATH" --subtensor.chain_endpoint "$CHAIN"

echo ""
echo "==> Localnet setup complete!"
echo "    Wallet path: $WALLET_PATH"
echo "    Chain:       $CHAIN"
echo "    Netuid:      $NETUID  (TPN subnet, Alice as owner)"
echo "    Validator:   bob     (Bob dev account)"
echo "    Miner:       charlie (Charlie dev account)"
