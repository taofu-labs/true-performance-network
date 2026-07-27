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
    from bittensor import Subtensor
    from bittensor import Wallet


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
    import bittensor_core
    from bittensor import calls
    from bittensor.intents.plan import Policy

    logger.info(
        f"timelocked_commit | hotkey={wallet.hotkey.ss58_address[:12]} "
        f"| blocks_until_reveal={blocks_until_reveal}"
    )

    # bittensor.timelock.encrypt() wraps the ciphertext in a SCALE `UserData`
    # envelope meant for other consumers; pallet-commitments expects the raw
    # compressed TLE ciphertext, so it silently fails to decode and the
    # commitment is dropped with no reveal. get_encrypted_commitment produces
    # the unwrapped ciphertext the pallet actually expects.
    ciphertext, reveal_round = bittensor_core.get_encrypted_commitment(
        reveal_payload, blocks_until_reveal, block_time
    )
    logger.debug(
        f"Encrypted commit | blocks_until_reveal={blocks_until_reveal} "
        f"| block_time={block_time} | reveal_round={reveal_round} "
        f"| ciphertext_len={len(ciphertext)} | payload={reveal_payload}"
    )
    info = {
        "fields": [[{
            "TimelockEncrypted": {"encrypted": ciphertext, "reveal_round": reveal_round}
        }]]
    }
    call = calls.Commitments.set_commitment(netuid, info)

    # submit_call is policy-gated; temporarily allow raw calls on this connection.
    original_policy = subtensor.policy
    subtensor.policy = Policy(allow_raw_calls=True)
    try:
        result = subtensor.submit_call(call, wallet, signer="hotkey")
        logger.debug(f"submit_call result: success={result.success} extrinsic_id={getattr(result, 'extrinsic_id', None)}")
        return result.success
    except Exception as e:
        logger.error(f"timelocked_commit failed: {e}")
        return False
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
