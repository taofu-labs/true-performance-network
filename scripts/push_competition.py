#!/usr/bin/env python3
"""
Create or update one competition on a leader validator from a single spec file.

Usage:
    ADMIN_API_KEY=... python scripts/push_competition.py \\
        --leader-url https://leader.tpn.internal \\
        competitions/tpn-001.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Path to a competition spec JSON file")
    parser.add_argument("--leader-url", required=True, help="Leader validator base URL")
    args = parser.parse_args()

    api_key = os.environ.get("ADMIN_API_KEY")
    if not api_key:
        print("ADMIN_API_KEY env var is required", file=sys.stderr)
        return 1

    spec = json.loads(args.file.read_text())
    resp = requests.post(
        f"{args.leader_url.rstrip('/')}/v1/competitions",
        json=spec,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=10,
    )
    if not resp.ok:
        print(f"failed: {resp.status_code} {resp.text}", file=sys.stderr)
        return 1

    print(f"pushed {resp.json()['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
