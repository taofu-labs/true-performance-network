"""HuggingFace repo accessibility check."""
import re
import requests

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
