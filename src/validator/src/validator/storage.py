"""
Persistent storage for validator state.

All data stored under ~/.tpn/validator-storage/ (POSIX) or
%APPDATA%/tpn/validator-storage/ (Windows).

Writes are atomic: data written to temp file in same directory,
then renamed over target — prevents truncated JSON on crash.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Iterator


def tpn_home() -> Path:
    """Platform-aware base dir: ~/.tpn on POSIX, %APPDATA%/tpn on Windows."""
    if os.name == "nt":
        return Path(os.environ.get("APPDATA", Path.home())) / "tpn"
    return Path.home() / ".tpn"


def validator_storage_dir() -> Path:
    """Returns ~/.tpn/validator-storage/, creating it if necessary."""
    d = tpn_home() / "validator-storage"
    d.mkdir(parents=True, exist_ok=True)
    return d


def clear_validator_storage() -> None:
    """Delete all .json files inside validator_storage_dir()."""
    d = validator_storage_dir()
    for p in d.glob("*.json"):
        p.unlink(missing_ok=True)


class PersistentSet:
    """String set backed by a single JSON file. Atomic writes. Auto-loads on init."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._data: set[str] = set()
        self.load()

    def load(self) -> None:
        """Read state from disk. Called automatically at construction."""
        if self._path.exists():
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(raw, list):
                    self._data = set(str(x) for x in raw)
                    return
            except (json.JSONDecodeError, OSError):
                pass
        self._data = set()

    def _save(self) -> None:
        """Atomically write current in-memory state to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(sorted(self._data), indent=2)
        fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, self._path)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def add(self, item: str) -> None:
        if item not in self._data:
            self._data.add(item)
            self._save()

    def discard(self, item: str) -> None:
        if item in self._data:
            self._data.discard(item)
            self._save()

    def clear(self) -> None:
        self._data.clear()
        self._save()

    def __contains__(self, item: object) -> bool:
        return item in self._data

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)
