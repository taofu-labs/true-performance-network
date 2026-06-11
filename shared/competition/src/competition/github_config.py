"""
Fetches competition specs from GitHub over HTTPS, or from local filesystem.

Always configured via a direct path/URL to index.json:
  https://raw.githubusercontent.com/.../competitions/index.json
  /abs/path/to/competitions/index.json

Individual competition files are resolved relative to the index location.
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional
import requests
from loguru import logger

from common.models.competition import CompetitionSpec

_SAFE_FILENAME_RE = re.compile(r"^[A-Za-z0-9_-]+\.json$")
_CACHE_TTL = 600

_cache: Dict[str, Dict[str, CompetitionSpec]] = {}
_cache_time: Dict[str, float] = {}


def _base_dir(index_url: str) -> str:
    return index_url.rsplit("/", 1)[0]


def get_all_competitions(
    index_url: str,
    force_refresh: bool = False,
) -> List[CompetitionSpec]:
    """Fetch all competitions listed in index.json. Results cached per index_url."""
    age = time.time() - _cache_time.get(index_url, 0.0)
    if not force_refresh and _cache.get(index_url) and age < _CACHE_TTL:
        return list(_cache[index_url].values())

    index = _fetch_index(index_url)
    if not index:
        cached = _cache.get(index_url)
        if cached:
            logger.warning("Competition fetch failed, using cached competition list")
            return list(cached.values())
        return []

    new_cache: Dict[str, CompetitionSpec] = {}
    for filename in index.get("competitions", []):
        if not isinstance(filename, str) or not _SAFE_FILENAME_RE.match(filename):
            logger.warning(f"Skipping unsafe competition filename: {filename!r}")
            continue
        spec = _fetch_competition(index_url, filename)
        if spec:
            new_cache[spec.id] = spec

    if new_cache:
        _cache[index_url] = new_cache
        _cache_time[index_url] = time.time()
        logger.info(f"Loaded {len(new_cache)} competitions")

    return list(_cache.get(index_url, {}).values())


def get_active_competitions(
    index_url: str,
    current_block: int,
    force_refresh: bool = False,
) -> List[CompetitionSpec]:
    """Return all competitions active at the given block (OPEN or SCORING phase)."""
    competitions = get_all_competitions(index_url=index_url, force_refresh=force_refresh)
    return [c for c in competitions if c.is_active(current_block)]


def get_competition_by_id(
    index_url: str,
    competition_id: str,
) -> Optional[CompetitionSpec]:
    """Fetch a specific competition by ID, using cache if available."""
    competitions = get_all_competitions(index_url=index_url)
    return next((c for c in competitions if c.id == competition_id), None)


def is_competition(index_url: str, competition_id: str) -> bool:
    """Return True if a competition with the given ID exists."""
    return get_competition_by_id(index_url=index_url, competition_id=competition_id) is not None


def _fetch_index(index_url: str) -> Optional[dict]:
    try:
        data = _load(index_url)
        if data.get("schema_version") != 1:
            logger.error(f"Unknown index schema version: {data.get('schema_version')}")
            return None
        return data
    except Exception as e:
        logger.error(f"Failed to fetch competition index from {index_url}: {e}")
        return None


def _fetch_competition(index_url: str, filename: str) -> Optional[CompetitionSpec]:
    url = f"{_base_dir(index_url)}/{filename}"
    try:
        data = _load(url)
        return CompetitionSpec.model_validate(data)
    except Exception as e:
        logger.error(f"Failed to fetch or parse competition {filename}: {e}")
        return None


def _load(url: str) -> dict:
    """Load JSON from a local path or an HTTP(S) URL."""
    if url.startswith("http://") or url.startswith("https://"):
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    with open(Path(url)) as f:
        return json.load(f)
