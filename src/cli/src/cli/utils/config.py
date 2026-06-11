import os
import json
from pathlib import Path


def tpn_home() -> Path:
    """Platform-aware base dir: ~/.tpn on POSIX, %APPDATA%/tpn on Windows."""
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "tpn"
    return Path.home() / ".tpn"


def identity_dir(coldkey: str, hotkey: str) -> Path:
    return tpn_home() / coldkey / hotkey


def init_identity_dir(coldkey: str, hotkey: str) -> Path:
    """Create ~/.tpn/<coldkey>/<hotkey>/ — called on registration."""
    d = identity_dir(coldkey, hotkey)
    d.mkdir(parents=True, exist_ok=True)
    return d


def competition_config_path(coldkey: str, hotkey: str, competition_id: str) -> Path:
    return identity_dir(coldkey, hotkey) / f"{competition_id}.json"


def load_competition_config(coldkey: str, hotkey: str, competition_id: str) -> dict:
    p = competition_config_path(coldkey, hotkey, competition_id)
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def save_competition_config(coldkey: str, hotkey: str, competition_id: str, data: dict) -> None:
    p = competition_config_path(coldkey, hotkey, competition_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))
    p.chmod(0o600)
