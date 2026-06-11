"""
Ban list management and lying detection.
Validator-side concern only — not part of competition logic.

TODO: Persist bans on-chain so all validators enforce the same list.
Bans are persisted to ~/.tpn/validator-storage/bans.json.
"""
from loguru import logger

from validator.storage import PersistentSet, validator_storage_dir

_BANNED: PersistentSet = PersistentSet(validator_storage_dir() / "bans.json")


def load_initial_bans(hotkeys: list):
    for hotkey in hotkeys:
        _BANNED.add(hotkey)
    if hotkeys:
        logger.info(f"Loaded {len(hotkeys)} banned hotkeys")


def is_banned(hotkey: str) -> bool:
    return hotkey in _BANNED


def ban(hotkey: str, reason: str):
    if hotkey in _BANNED:
        return
    _BANNED.add(hotkey)
    logger.warning(f"[BAN] {hotkey[:12]}... : {reason}")


def check_lying(
    hotkey: str,
    reveal_size: int,
    actual_size: int,
    reveal_sha256: str,
    actual_sha256: str,
    tolerance: float = 0.02,
) -> bool:
    """
    Detects lying on verifiable fields (size and SHA256).
    Score lying cannot be detected until real eval replaces the stub.
    Returns True and bans the hotkey if lying is found.
    """
    if actual_sha256 != reveal_sha256:
        ban(hotkey, f"SHA256 mismatch: revealed={reveal_sha256[:12]}, actual={actual_sha256[:12]}")
        return True
    if reveal_size != actual_size:
        ban(hotkey, f"size mismatch: revealed={reveal_size}, actual={actual_size}")
        return True
    return False
