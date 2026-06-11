"""
Chain interaction helpers for TPN.
Used by both the CLI (miner) and the validator.

bittensor is imported lazily (inside functions) to prevent its arg-parser
from hijacking sys.argv when the CLI loads.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
from loguru import logger
import tenacity

from common import settings as common_settings

if TYPE_CHECKING:
    from bittensor.core.subtensor import Subtensor
    from bittensor_wallet import Wallet


def _log_subtensor_retry(retry_state):
    logger.warning(f"Retry attempt {retry_state.attempt_number} connecting to subtensor")


@tenacity.retry(
    stop=tenacity.stop_after_attempt(3),
    wait=tenacity.wait_exponential(multiplier=1, min=4, max=60),
    before_sleep=_log_subtensor_retry,
)
def get_subtensor(network: Optional[str] = None) -> "Subtensor":
    """
    Return a Subtensor instance.
    Uses network arg if provided, else common_settings.NETWORK.
    Raises if BITTENSOR=False in settings.
    """
    if not common_settings.BITTENSOR:
        raise Exception("BITTENSOR=False — subtensor disabled")
    from bittensor.core.subtensor import Subtensor
    target = network or common_settings.NETWORK
    logger.info(f"Connecting to subtensor: {target}")
    return Subtensor(network=target)


def get_wallet(
    coldkey: str,
    hotkey: str,
    wallet_path: Optional[str] = None,
) -> "Wallet":
    """Return a Wallet instance for the given coldkey/hotkey."""
    from bittensor_wallet import Wallet
    kwargs = {"name": coldkey, "hotkey": hotkey}
    if wallet_path:
        kwargs["path"] = wallet_path
    return Wallet(**kwargs)


def timelocked_commit(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    reveal_payload: str,
    blocks_until_reveal: int,
    block_time: float = 12.0,
) -> bool:
    """
    Encrypt reveal_payload with TLE and submit to chain.
    The chain auto-decrypts at the corresponding drand round and stores
    the plaintext in RevealedCommitments storage.
    """
    logger.info(
        f"timelocked_commit | hotkey={wallet.hotkey.ss58_address[:12]} "
        f"| blocks_until_reveal={blocks_until_reveal}"
    )
    result = subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=netuid,
        data=reveal_payload,
        blocks_until_reveal=blocks_until_reveal,
        block_time=block_time,
    )
    return result.success


def is_hotkey_registered(
    subtensor: "Subtensor",
    hotkey_ss58: str,
    netuid: int,
) -> bool:
    """Returns True if hotkey is registered on netuid."""
    try:
        return subtensor.is_hotkey_registered(hotkey_ss58=hotkey_ss58, netuid=netuid)
    except Exception as e:
        logger.warning(f"is_hotkey_registered check failed: {e}")
        return False


def _strip_scale_prefix(s: str) -> str:
    """
    Strip the SCALE compact-length prefix baked into revealed commitment strings.
    The substrate client SCALE-encodes the BoundedVec<u8> encrypted field before
    TLE encryption, so the compact prefix is encrypted in and decrypted back out,
    appearing as the first 1-4 characters of the revealed string.
    """
    if not s:
        return s
    mode = ord(s[0]) & 0b11
    offset = 1 if mode == 0 else (2 if mode == 1 else 4)
    return s[offset:]


def read_revealed_commitments(
    subtensor: "Subtensor",
    netuid: int,
) -> dict[str, list[tuple[str, int]]]:
    """
    Returns {hotkey_ss58: [(plaintext_str, reveal_block), ...]} for all miners.
    Reads RevealedCommitments storage directly via substrate query_map, bypassing
    the SDK decoder which incorrectly assumes hex encoding.
    """
    try:
        raw = subtensor.substrate.query_map(
            "Commitments", "RevealedCommitments", params=[netuid]
        )
        result: dict[str, list[tuple[str, int]]] = {}
        for hotkey, entries in raw:
            decoded = []
            for raw_str, block in entries:
                try:
                    decoded.append((_strip_scale_prefix(raw_str), int(block)))
                except Exception as e:
                    logger.debug(f"Failed to decode reveal for {hotkey[:12]}: {e}")
            if decoded:
                result[hotkey] = decoded
        return result
    except Exception as e:
        logger.warning(f"read_revealed_commitments failed: {e}")
        return {}
