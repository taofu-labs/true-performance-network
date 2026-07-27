#!/usr/bin/env python3
"""
One-time migration: POST every competitions/*.json file to a leader validator's
admin endpoint, seeding its DB from the old git-hosted configs.

Usage:
    ADMIN_API_KEY=... python scripts/seed_competitions.py \\
        --leader-url https://leader.tpn.internal \\
        --dir competitions
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--leader-url", required=True, help="Leader validator base URL")
    parser.add_argument("--dir", default="competitions", help="Directory containing index.json + spec files")
    args = parser.parse_args()

    api_key = os.environ.get("ADMIN_API_KEY")
    if not api_key:
        print("ADMIN_API_KEY env var is required", file=sys.stderr)
        return 1

    comp_dir = Path(args.dir)
    index = json.loads((comp_dir / "index.json").read_text())

    for filename in index["competitions"]:
        spec = json.loads((comp_dir / filename).read_text())
        resp = requests.post(
            f"{args.leader_url.rstrip('/')}/v1/competitions",
            json=spec,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if resp.ok:
            print(f"seeded {spec['id']}")
        else:
            print(f"failed {filename}: {resp.status_code} {resp.text}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
