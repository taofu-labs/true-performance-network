#!/usr/bin/env python3
"""
Run the leader's /admin competition page locally, without a chain or validator wallet.

Uses the real API handlers and signature checks; only the chain is faked (block number
advances every 12s from 1000) and the leader hotkey is a throwaway key. Created
competitions go to a SQLite file in the temp dir (override with --db).

Usage:
    uv run --package validator python scripts/admin_page_dev.py --signers 5Abc...,5Def...
    open http://localhost:9200/admin
"""
import argparse
import tempfile
import time
from pathlib import Path

from aiohttp import web
from bittensor import sp_core

from validator import settings as validator_settings
from validator import store
from validator.api import LeaderApiMixin, _require_admin_signer


class FakeSubtensor:
    def __init__(self):
        self.started = time.time()

    def block(self):
        return 1000 + int((time.time() - self.started) / 12)


class DevLeader(LeaderApiMixin):
    def __init__(self, db_path):
        self._db = store.init_db(db_path)
        self.subtensor = FakeSubtensor()
        self.hotkey = sp_core.Keypair.create_from_mnemonic(
            sp_core.Keypair.generate_mnemonic(12), sp_core.CRYPTO_SR25519
        ).ss58_address


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signers", required=True, help="comma-separated ss58 addresses")
    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--db", default=str(Path(tempfile.gettempdir()) / "tpn_admin_page_dev.db"))
    args = parser.parse_args()

    validator_settings.ADMIN_SIGNERS = {s.strip() for s in args.signers.split(",") if s.strip()}
    leader = DevLeader(args.db)
    app = web.Application()
    app.router.add_get("/admin", leader._h_admin_page)
    app.router.add_get("/v1/admin/head", leader._h_admin_head)
    app.router.add_get("/v1/competitions", leader._h_list_competitions)
    app.router.add_get("/v1/competitions/{competition_id}", leader._h_get_competition)
    app.router.add_post(
        "/v1/admin/competitions", _require_admin_signer(leader._h_create_competition, leader.hotkey)
    )
    print(f"Open http://localhost:{args.port}/admin  (db: {args.db}, signers: {', '.join(validator_settings.ADMIN_SIGNERS)})")
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
