#!/usr/bin/env bash
set -euo pipefail

CHAIN="ws://127.0.0.1:9946"
NETUID=2

# Dev accounts — coldkeys are pre-funded at genesis on this chain spec via the
# standard substrate dev derivation (//Alice, //Bob, //Charlie).
OWNER="alice"      # subnet owner (Alice coldkey, pre-funded)
VALIDATOR="bob"    # validator
MINER="charlie"    # miner

# Repo-local wallet dir — keeps localnet keys out of ~/.bittensor
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WALLET_PATH="$SCRIPT_DIR/../wallets"
mkdir -p "$WALLET_PATH"

BTCLI=(uv run --project "$SCRIPT_DIR/.." btcli)
PYTHON=(uv run --project "$SCRIPT_DIR/.." python3)

# ── Wallet creation ────────────────────────────────────────────────────────────
# bittensor 11's CLI dropped --uri (SURI derivation), so coldkeys can't be
# regenerated through btcli itself anymore. bittensor_core.Keypair still
# implements the derivation though (just not wired to the CLI) — use it
# directly to recreate the coldkey files if missing. Hotkeys are NOT derived
# from a URI (they're random in this script, like upstream `wallet new_hotkey`)
# so a missing hotkey just gets a fresh one + re-registration, no recovery needed.

echo "==> Checking dev wallets in $WALLET_PATH ..."

recreate_coldkey() {
  local name="$1" suri="$2"
  mkdir -p "$WALLET_PATH/$name"
  "${PYTHON[@]}" -c "
import json
from bittensor_core import Keypair, serialized_keypair_to_keyfile_data

kp = Keypair.create_from_uri('//$suri')
full = json.loads(serialized_keypair_to_keyfile_data(kp))
pub = {k: v for k, v in full.items() if k != 'privateKey'}

with open('$WALLET_PATH/$name/coldkey', 'w') as f:
    json.dump(full, f)
with open('$WALLET_PATH/$name/coldkeypub.txt', 'w') as f:
    json.dump(pub, f)

print(f'    Recreated {\"$name\"} coldkey from //$suri -> {kp.ss58_address}')
"
}

SURI_alice="Alice"
SURI_bob="Bob"
SURI_charlie="Charlie"

for name in "$OWNER" "$VALIDATOR" "$MINER"; do
  if [[ -f "$WALLET_PATH/$name/coldkeypub.txt" ]]; then
    echo "    '$name' coldkey already exists, skipping"
  else
    suri_var="SURI_${name}"
    echo "==> Recreating '$name' coldkey from //${!suri_var} (deterministic, genesis-funded)..."
    recreate_coldkey "$name" "${!suri_var}"
  fi

  if [[ -f "$WALLET_PATH/$name/hotkeys/default" ]]; then
    echo "    '$name' hotkey already exists, skipping"
  else
    echo "==> Creating fresh hotkey for '$name' (not recoverable — random, will need (re)registration)..."
    "${BTCLI[@]}" wallet new-hotkey \
      --wallet "$name" \
      --wallet-path "$WALLET_PATH" \
      --wallet-hotkey default \
      --n-words 12 \
      --quiet
  fi
done

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
# subnet create no longer accepts identity flags (--subnet-name etc) — identity
# is set separately via `sudo set-identity` after creation.

if "${BTCLI[@]}" subnet list --network "$CHAIN" --json 2>/dev/null | python3 -c "
import json, sys
subs = json.load(sys.stdin)
sys.exit(0 if any(s.get('name') == 'TPN Localnet' for s in subs) else 1)
"; then
  echo "==> TPN subnet already exists, skipping creation"
else
  echo "==> Creating TPN subnet (Alice as owner)..."
  "${BTCLI[@]}" subnet create \
    --wallet "$OWNER" \
    --wallet-path "$WALLET_PATH" \
    --wallet-hotkey default \
    --network "$CHAIN" \
    --yes

  # Netuid of the subnet just created is the highest netuid now on chain.
  NETUID=$("${BTCLI[@]}" subnet list --network "$CHAIN" --json | python3 -c "
import json, sys
subs = json.load(sys.stdin)
print(max(s['netuid'] for s in subs))
")

  echo "==> Setting subnet identity on netuid $NETUID..."
  "${BTCLI[@]}" sudo set-identity \
    --netuid "$NETUID" \
    --wallet "$OWNER" \
    --wallet-path "$WALLET_PATH" \
    --wallet-hotkey default \
    --network "$CHAIN" \
    --name "TPN Localnet" \
    --url "https://github.com/taofu-labs/tao-performance-network" \
    --description "TPN localnet dev subnet" \
    --yes
fi

# Resolve netuid by name every run (survives reruns/duplicate-creation drift).
NETUID=$("${BTCLI[@]}" subnet list --network "$CHAIN" --json | python3 -c "
import json, sys
subs = json.load(sys.stdin)
matches = [s['netuid'] for s in subs if s.get('name') == 'TPN Localnet']
print(max(matches))
")
echo "==> Using netuid $NETUID"

# ── Helper: check if hotkey is registered on NETUID ───────────────────────────

is_registered() {
  local wallet="$1"
  local hotkey_ss58
  hotkey_ss58=$(python3 -c "import json; print(json.load(open('$WALLET_PATH/$wallet/hotkeys/default'))['ss58Address'])" 2>/dev/null) || return 1

  "${BTCLI[@]}" subnet metagraph "$NETUID" --network "$CHAIN" --json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if '$hotkey_ss58' in d.get('hotkeys', []) else 1)
"
}

# ── Register validator ─────────────────────────────────────────────────────────

if is_registered "$VALIDATOR"; then
  echo "==> Validator (Bob) already registered on netuid $NETUID, skipping"
else
  echo "==> Registering validator (Bob) on netuid $NETUID..."
  "${BTCLI[@]}" subnet register \
    --wallet "$VALIDATOR" \
    --wallet-path "$WALLET_PATH" \
    --wallet-hotkey default \
    --netuid "$NETUID" \
    --network "$CHAIN" \
    --yes
fi

# ── Register miner ─────────────────────────────────────────────────────────────

if is_registered "$MINER"; then
  echo "==> Miner (Charlie) already registered on netuid $NETUID, skipping"
else
  echo "==> Registering miner (Charlie) on netuid $NETUID..."
  "${BTCLI[@]}" subnet register \
    --wallet "$MINER" \
    --wallet-path "$WALLET_PATH" \
    --wallet-hotkey default \
    --netuid "$NETUID" \
    --network "$CHAIN" \
    --yes \
    --quiet
fi

# ── Start subnet emissions (enables staking) ──────────────────────────────────

echo "==> Checking if subnet $NETUID can start..."
"${BTCLI[@]}" sudo check-start --netuid "$NETUID" --network "$CHAIN"

echo "==> Starting subnet $NETUID emissions (Alice as owner)..."
"${BTCLI[@]}" sudo start \
  --wallet "$OWNER" \
  --wallet-path "$WALLET_PATH" \
  --wallet-hotkey default \
  --netuid "$NETUID" \
  --network "$CHAIN" \
  --yes || echo "    (already started — ignoring)"

# ── Stake validator ────────────────────────────────────────────────────────────

echo "==> Adding stake to validator (Bob) on netuid $NETUID..."
"${BTCLI[@]}" stake add \
  --wallet "$VALIDATOR" \
  --wallet-path "$WALLET_PATH" \
  --wallet-hotkey default \
  --network "$CHAIN" \
  --netuid "$NETUID" \
  --amount-tao 10 \
  --no-slippage-protection \
  --no-mev-shield \
  --yes

# ── Verify ─────────────────────────────────────────────────────────────────────

echo "==> Verifying setup..."
"${BTCLI[@]}" subnet list --network "$CHAIN"
"${BTCLI[@]}" wallet overview --wallet "$VALIDATOR" --wallet-path "$WALLET_PATH" --network "$CHAIN"
"${BTCLI[@]}" wallet overview --wallet "$MINER"     --wallet-path "$WALLET_PATH" --network "$CHAIN"

echo ""
echo "==> Localnet setup complete!"
echo "    Wallet path: $WALLET_PATH"
echo "    Chain:       $CHAIN"
echo "    Netuid:      $NETUID  (TPN subnet, Alice as owner)"
echo "    Validator:   bob     (Bob dev account)"
echo "    Miner:       charlie (Charlie dev account)"
