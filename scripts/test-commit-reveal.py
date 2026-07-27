#!/usr/bin/env python3
"""
Manual test: submit a timelocked commit against the local devnet and poll
until it reveals. Standalone — not wired into the validator/CLI packages.

Usage:
    uv run --project src/validator python scripts/test-commit-reveal.py \
        [--wallet charlie] [--netuid 2] [--network ws://localhost:9946] \
        [--reveal-in 5] [--timeout 120]
"""
import argparse
import time

import bittensor as bt
import bittensor_core
from bittensor import calls
from bittensor.intents.plan import Policy


def main():
    parser = argparse.ArgumentParser(description="Test commit-reveal against local chain")
    parser.add_argument("--wallet", default="charlie", help="Wallet name (coldkey)")
    parser.add_argument("--hotkey", default="default", help="Hotkey name")
    parser.add_argument("--wallet-path", default="./wallets", help="Wallet directory")
    parser.add_argument("--netuid", type=int, default=2)
    parser.add_argument("--network", default="ws://localhost:9946")
    parser.add_argument("--payload", default="commit-reveal-test-payload")
    parser.add_argument("--reveal-in", type=float, default=5.0, help="Seconds until reveal")
    parser.add_argument("--timeout", type=float, default=120.0, help="Max seconds to poll for reveal")
    parser.add_argument("--poll-interval", type=float, default=3.0)
    args = parser.parse_args()

    print(f"Connecting to {args.network} ...")
    subtensor = bt.Subtensor(network=args.network, policy=Policy(allow_raw_calls=True))
    wallet = bt.Wallet(name=args.wallet, hotkey=args.hotkey, path=args.wallet_path)
    hotkey_ss58 = wallet.hotkey.ss58_address
    print(f"Wallet hotkey: {hotkey_ss58}")

    uid = subtensor.neurons.uid(hotkey_ss58, args.netuid)
    if uid is None:
        print(f"ERROR: hotkey not registered on netuid {args.netuid}. Run setup-localnet.sh first.")
        raise SystemExit(1)
    print(f"Registered as uid {uid} on netuid {args.netuid}")

    # bt.timelock.encrypt() wraps the ciphertext in a SCALE `UserData` envelope
    # that pallet-commitments can't decode (it expects a raw compressed TLE
    # ciphertext), silently dropping the commitment with no reveal.
    # get_encrypted_commitment produces the unwrapped bytes the pallet expects.
    blocks_until_reveal = max(1, round(args.reveal_in / 12.0))
    ciphertext, reveal_round = bittensor_core.get_encrypted_commitment(
        args.payload, blocks_until_reveal, 12.0
    )
    print(f"Sealed payload: {len(ciphertext)} bytes, reveal_round={reveal_round}")

    info = {
        "fields": [[{
            "TimelockEncrypted": {"encrypted": ciphertext, "reveal_round": reveal_round}
        }]]
    }
    call = calls.Commitments.set_commitment(args.netuid, info)

    print("Submitting commit...")
    result = subtensor.submit_call(call, wallet, signer="hotkey")
    if not result.success:
        print(f"ERROR: submission failed: {result}")
        raise SystemExit(1)
    print(f"Submitted OK. extrinsic_id={result.extrinsic_id}")

    print(f"Polling for reveal (timeout={args.timeout}s, every {args.poll_interval}s)...")
    start = time.monotonic()
    while time.monotonic() - start < args.timeout:
        commitments = subtensor.identity.commitments(args.netuid)
        entry = next((c for c in commitments if c["hotkey"] == hotkey_ss58), None)
        if entry is None:
            print("  no commitment entry found yet")
        else:
            elapsed = time.monotonic() - start
            print(f"  [{elapsed:5.1f}s] status={entry['status']} is_revealed={entry['is_revealed']} "
                  f"reveals_at={entry.get('reveals_at')}")
            if entry["is_revealed"]:
                revealed = subtensor.identity.revealed_commitment(args.netuid, hotkey_ss58)
                print(f"\nREVEALED after {elapsed:.1f}s")
                print(f"revealed_commitment(): {revealed}")  # list of (plaintext_bytes, reveal_block)
                if revealed and any(args.payload.encode() in p for p, _ in revealed):
                    print("Payload matches — round trip OK.")
                else:
                    print("WARNING: payload not found in revealed data — check encoding/chunking.")
                return
        time.sleep(args.poll_interval)

    print(f"\nTIMEOUT after {args.timeout}s — never revealed. "
          f"Check whether the local chain image has drand-beacon connectivity.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
