import argparse
import asyncio

from validator import settings as validator_settings
from validator.validator import Validator
import validator.storage as storage


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TPN Validator")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Wipe persistent validator storage (bans, scored) before starting.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.clean:
        storage.clear_validator_storage()
        print(f"Cleared validator storage at {storage.validator_storage_dir()}")

    validator = Validator(
        coldkey=validator_settings.WALLET_COLDKEY,
        wallet_hotkey=validator_settings.WALLET_HOTKEY,
    )
    asyncio.run(validator.run_validator())
