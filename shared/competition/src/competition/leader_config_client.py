"""
Fetches competition specs from a leader validator's public HTTP API.

Drop-in replacement for the removed github_config.py — same function names/shapes,
but keyed by a leader validator base_url (e.g. https://val0.trueperformancenetwork.com) instead
of a path/URL to index.json.
"""
import time
from typing import Dict, List, Optional
import requests
from loguru import logger

from common.models.competition import CompetitionSpec

_CACHE_TTL = 600

_cache: Dict[str, Dict[str, CompetitionSpec]] = {}
_cache_time: Dict[str, float] = {}


def get_all_competitions(
    base_url: str,
    force_refresh: bool = False,
) -> List[CompetitionSpec]:
    """Fetch all competitions from the leader. Results cached per base_url."""
    age = time.time() - _cache_time.get(base_url, 0.0)
    if not force_refresh and _cache.get(base_url) and age < _CACHE_TTL:
        return list(_cache[base_url].values())

    specs = _fetch_all(base_url)
    if specs is None:
        cached = _cache.get(base_url)
        if cached:
            logger.warning("Competition fetch failed, using cached competition list")
            return list(cached.values())
        return []

    new_cache = {spec.id: spec for spec in specs}
    _cache[base_url] = new_cache
    _cache_time[base_url] = time.time()
    logger.info(f"Loaded {len(new_cache)} competitions")
    return list(new_cache.values())


def get_active_competitions(
    base_url: str,
    current_block: int,
    force_refresh: bool = False,
) -> List[CompetitionSpec]:
    """Return all competitions active at the given block (OPEN, SCORING, or post-scoring distribution phase)."""
    competitions = get_all_competitions(base_url=base_url, force_refresh=force_refresh)
    return [c for c in competitions if c.is_active(current_block)]


def get_competition_by_id(
    base_url: str,
    competition_id: str,
) -> Optional[CompetitionSpec]:
    """Fetch a specific competition by ID, using cache if available."""
    competitions = get_all_competitions(base_url=base_url)
    return next((c for c in competitions if c.id == competition_id), None)


def is_competition(base_url: str, competition_id: str) -> bool:
    """Return True if a competition with the given ID exists."""
    return get_competition_by_id(base_url=base_url, competition_id=competition_id) is not None


def _fetch_all(base_url: str) -> Optional[List[CompetitionSpec]]:
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/v1/competitions", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [CompetitionSpec.model_validate(item) for item in data.get("competitions", [])]
    except Exception as e:
        logger.error(f"Failed to fetch competitions from {base_url}: {e}")
        return None
