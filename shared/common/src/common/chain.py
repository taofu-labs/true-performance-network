"""
Chain interaction helpers for TPN.
Used by both the CLI (miner) and the validator.

bittensor is imported lazily (inside functions) to prevent its arg-parser
from hijacking sys.argv when the CLI loads.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from loguru import logger
import tenacity

from common import settings as common_settings

if TYPE_CHECKING:
    from bittensor import Subtensor
    from bittensor import Wallet


@dataclass
class TimelockedCommitResult:
    """Result of a timelocked_commit call."""

    success: bool
    reveal_round: int


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
    from bittensor import Subtensor
    target = network or common_settings.NETWORK
    logger.info(f"Connecting to subtensor: {target}")
    return Subtensor(network=target)


def get_wallet(
    coldkey: str,
    hotkey: str,
    wallet_path: Optional[str] = None,
) -> "Wallet":
    """Return a Wallet instance for the given coldkey/hotkey."""
    from bittensor import Wallet
    kwargs = {"name": coldkey, "hotkey": hotkey}
    if wallet_path:
        kwargs["path"] = wallet_path
    return Wallet(**kwargs)


MAX_COMMITMENT_FIELDS = 3  # pallet-commitments MaxFields (runtime MaxCommitFieldsInner)


def timelocked_commit(
    subtensor: "Subtensor",
    wallet: "Wallet",
    netuid: int,
    reveal_payload: str,
    blocks_until_reveal: int,
    block_time: float = 12.0,
    previous_reveal_round: Optional[int] = None,
) -> "TimelockedCommitResult":
    """
    Encrypt reveal_payload with TLE and submit to chain.
    The chain auto-decrypts at the corresponding drand round and stores
    the plaintext in RevealedCommitments storage.

    set_commitment REPLACES the whole CommitmentOf.info.fields list (capped at
    MAX_COMMITMENT_FIELDS), so a naive single-field call would silently drop
    any other pending (not-yet-revealed) commitment for this hotkey/subnet —
    e.g. an overlapping competition committed earlier. To keep those alive we
    read the existing fields first and append.

    If `previous_reveal_round` is given and matches the reveal_round of an
    existing pending field, that field is replaced in place instead of
    appended — this is the republish-same-competition case (caller updating
    their submission before it reveals) and must not count twice against
    MAX_COMMITMENT_FIELDS or produce two reveals for the same competition.
    """
    import bittensor_core
    from bittensor import calls
    from bittensor._generated import storage as st
    from bittensor.intents.plan import Policy

    logger.info(
        f"timelocked_commit | hotkey={wallet.hotkey.ss58_address[:12]} "
        f"| blocks_until_reveal={blocks_until_reveal}"
    )

    existing = subtensor.query(st.Commitments.CommitmentOf, [netuid, wallet.hotkey.ss58_address])
    existing_fields = list((existing.get("info") or {}).get("fields") or []) if existing else []

    def _reveal_round(field: dict) -> Optional[int]:
        return (field.get("TimelockEncrypted") or {}).get("reveal_round")

    replace_index = None
    if previous_reveal_round is not None:
        for i, field in enumerate(existing_fields):
            if _reveal_round(field) == previous_reveal_round:
                replace_index = i
                break

    if replace_index is None and len(existing_fields) >= MAX_COMMITMENT_FIELDS:
        raise ValueError(
            f"Cannot submit commit: hotkey already has {len(existing_fields)} pending "
            f"commitment(s) on subnet {netuid} (max {MAX_COMMITMENT_FIELDS}). "
            "Wait for an existing commit to reveal before submitting another."
        )

    ciphertext, reveal_round = bittensor_core.get_encrypted_commitment(
        reveal_payload, blocks_until_reveal, block_time
    )
    logger.debug(
        f"Encrypted commit | blocks_until_reveal={blocks_until_reveal} "
        f"| block_time={block_time} | reveal_round={reveal_round} "
        f"| ciphertext_len={len(ciphertext)} | payload={reveal_payload} "
        f"| existing_pending={len(existing_fields)} | replace_index={replace_index}"
    )
    new_field = {
        "TimelockEncrypted": {"encrypted": ciphertext, "reveal_round": reveal_round}
    }
    if replace_index is not None:
        fields = list(existing_fields)
        fields[replace_index] = new_field
    else:
        fields = existing_fields + [new_field]
    info = {"fields": fields}
    call = calls.Commitments.set_commitment(netuid, info)

    # submit_call is policy-gated; temporarily allow raw calls on this connection.
    original_policy = subtensor.policy
    subtensor.policy = Policy(allow_raw_calls=True)
    try:
        result = subtensor.submit_call(call, wallet, signer="hotkey")
        logger.debug(f"submit_call result: success={result.success} extrinsic_id={getattr(result, 'extrinsic_id', None)}")
        return TimelockedCommitResult(success=result.success, reveal_round=reveal_round)
    except Exception as e:
        logger.error(f"timelocked_commit failed: {e}")
        return TimelockedCommitResult(success=False, reveal_round=reveal_round)
    finally:
        subtensor.policy = original_policy


def is_hotkey_registered(
    subtensor: "Subtensor",
    hotkey_ss58: str,
    netuid: int,
) -> bool:
    """Returns True if hotkey is registered on netuid."""
    try:
        return subtensor.neurons.uid(hotkey_ss58, netuid) is not None
    except Exception as e:
        logger.warning(f"is_hotkey_registered check failed: {e}")
        return False


def read_revealed_commitments(
    subtensor: "Subtensor",
    netuid: int,
) -> dict[str, list[tuple[str, int]]]:
    """
    Returns {hotkey_ss58: [(plaintext_str, reveal_block), ...]} for all miners.
    Uses the metagraph's typed commitment map — decryption/decoding handled
    internally by the SDK, each NeuronCommitment.revealed already a
    (reveal_block, plaintext) history, oldest first.
    """
    try:
        metagraph = subtensor.subnets.metagraph(netuid)
        result: dict[str, list[tuple[str, int]]] = {}
        if metagraph is None:
            return result
        logger.debug(f"Read {len(metagraph.commitments)} commitments from metagraph: {list(metagraph.commitments.values())}")
        for commitment in metagraph.commitments.values():
            if commitment and commitment.revealed:
                result[commitment.hotkey] = [
                    (plaintext, block) for block, plaintext in commitment.revealed
                ]
        return result
    except Exception as e:
        logger.warning(f"read_revealed_commitments failed: {e}")
        return {}
