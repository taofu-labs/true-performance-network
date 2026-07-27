import argparse
import asyncio

from common.settings import configure_logging
from validator import settings as validator_settings
from validator.validator import Validator
import validator.store as store

configure_logging()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPN Validator")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe persistent validator storage (bans, scored competitions, scoring history) before starting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.clean:
        store.clear_validator_db()
        print(f"Cleared validator storage at {store.validator_db_path()}")

    validator = Validator(
        coldkey=validator_settings.WALLET_COLDKEY,
        wallet_hotkey=validator_settings.WALLET_HOTKEY,
    )
    asyncio.run(validator.run_validator())
