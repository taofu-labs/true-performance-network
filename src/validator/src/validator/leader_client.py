"""
Follower validator's HTTP client for the leader validator's read API.
"""
from typing import List, Optional

import requests


class LeaderClient:
    def __init__(self, base_url: str, timeout: int = 30):
        if not base_url:
            raise ValueError("LeaderClient requires a base_url (LEADER_VALIDATOR_URL)")
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def get_scoring_results(self, competition_id: str, since_run_id: int = 0) -> tuple[List[dict], Optional[str]]:
        """Returns (runs, scored_status). scored_status is 'scored', 'failed_no_reveals', or None if not yet scored."""
        resp = requests.get(
            f"{self._base}/v1/follower/scoring-results",
            params={"competition_id": competition_id, "since_run_id": since_run_id},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("runs", []), body.get("scored_status")
