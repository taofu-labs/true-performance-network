"""HuggingFace download and verification."""
import hashlib
import os
import re
from typing import Optional
import requests
from huggingface_hub import snapshot_download
from loguru import logger

# username/repo-name — only alphanumeric, hyphens, underscores, dots; no path traversal
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$")


def _validate_repo_id(repo_id: str) -> bool:
    """Reject repo IDs that don't look like 'owner/name'."""
    return bool(_REPO_ID_RE.match(repo_id))



def check_repo_public(repo_id: str) -> bool:
    if not _validate_repo_id(repo_id):
        return False
    try:
        resp = requests.head(
            f"https://huggingface.co/{repo_id}", timeout=10, allow_redirects=True
        )
        return resp.status_code == 200
    except Exception:
        return False


def download_repo(repo_id: str, local_dir: str) -> str:
    """Download a public HuggingFace repo. Returns local directory path."""
    if not _validate_repo_id(repo_id):
        raise ValueError(f"Invalid repo ID: {repo_id!r}")
    return snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        ignore_patterns=["*.md", ".gitattributes"],
    )


def find_gguf_file(local_dir: str) -> Optional[str]:
    for root, _, files in os.walk(local_dir):
        for f in files:
            if f.endswith(".gguf"):
                return os.path.join(root, f)
    return None


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_gguf_header(gguf_path: str) -> bool:
    """Check GGUF magic bytes and version. Does not load the model."""
    try:
        with open(gguf_path, "rb") as f:
            if f.read(4) != b"GGUF":
                return False
            version = int.from_bytes(f.read(4), "little")
            return version in (1, 2, 3)
    except Exception:
        return False
