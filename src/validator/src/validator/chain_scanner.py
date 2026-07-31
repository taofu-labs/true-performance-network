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
    results: Dict[str, MinerSubmission] = {}

    logger.debug(f"scan_reveals | {competition.id} | commit_end_block={competition.commit_end_block} | {len(all_reveals)} hotkeys with reveals: {all_reveals}")

    for hotkey, entries in all_reveals.items():
        if store.is_banned(conn, hotkey):
            logger.debug(f"Skipping banned hotkey {hotkey[:12]}")
            continue
        # Accept reveals landing within a small window after commit_end_block —
        # the chain's TLE auto-decrypt lands within ~5 blocks, not exactly on it.
        for raw, reveal_block in entries:
            if _matches_block(reveal_block, competition.commit_end_block):
                submission = parse_reveal_payload(raw)
                if submission is None:
                    logger.debug(f"Malformed reveal payload for {hotkey[:12]}")
                    continue
                if submission.competition_id != competition.id:
                    logger.debug(f"Wrong competition_id in reveal for {hotkey[:12]}")
                    continue
                logger.debug(f"Accepted submission from {hotkey[:12]}: {submission}")
                results[hotkey] = submission
                break  # one submission per hotkey
            else:
                logger.debug(f"reveal_block={reveal_block} outside window for commit_end_block={competition.commit_end_block} (hotkey={hotkey[:12]})")

    logger.info(f"Found {len(results)} valid reveals for competition {competition.id}")
    return results


def _matches_block(reveal_block: int, commit_end_block: int) -> bool:
    return commit_end_block <= reveal_block < commit_end_block + 5
