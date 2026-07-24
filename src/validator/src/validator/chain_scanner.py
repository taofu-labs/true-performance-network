"""
Reads auto-revealed commitment payloads from chain.

After commit_end_block the chain auto-decrypts TLE commitments and stores
plaintext in RevealedCommitments storage. Validators read that storage directly.
"""
from __future__ import annotations
import sqlite3
from typing import TYPE_CHECKING, Dict
from loguru import logger

if TYPE_CHECKING:
    from bittensor import Subtensor

from common.models.submission import MinerSubmission, parse_reveal_payload
from common.models.competition import CompetitionSpec
from common.chain import read_revealed_commitments
from validator import store
from common import settings


def scan_reveals(
    subtensor: Subtensor,
    competition: CompetitionSpec,
    conn: sqlite3.Connection,
) -> Dict[str, MinerSubmission]:
    """
    Read auto-revealed payloads from RevealedCommitments storage.
    Returns {hotkey: MinerSubmission} — one valid submission per hotkey (newest wins).
    """
    all_reveals = read_revealed_commitments(subtensor, settings.NETUID)
    is_fast_blocks = subtensor.is_fast_blocks()
    results: Dict[str, MinerSubmission] = {}

    for hotkey, entries in all_reveals.items():
        if store.is_banned(conn, hotkey):
            continue
        # Only accept reveals at the competition's commit_end_block
        for raw, reveal_block in entries:
            if _matches_block(reveal_block, competition.commit_end_block, is_fast_blocks):
                submission = parse_reveal_payload(raw)
                if submission is None:
                    logger.debug(f"Malformed reveal payload for {hotkey[:12]}")
                    continue
                if submission.competition_id != competition.id:
                    logger.debug(f"Wrong competition_id in reveal for {hotkey[:12]}")
                    continue
                results[hotkey] = submission
                break  # one submission per hotkey

    logger.info(f"Found {len(results)} valid reveals for competition {competition.id}")
    return results


def _matches_block(
    reveal_block: int,
    commit_end_block: int,
    is_fast_blocks: bool,
) -> bool:
    if is_fast_blocks: 
        return reveal_block > commit_end_block - 5 and reveal_block < commit_end_block + 5
    return reveal_block == commit_end_block
