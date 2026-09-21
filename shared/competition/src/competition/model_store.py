"""HuggingFace repo accessibility check."""
import re
import requests

# username/repo-name — alphanumeric, hyphens, underscores and dots. Dots are
# common in real repo names (Qwen2.5, Llama-3.1, Phi-3.5), so excluding them
# rejected most modern models as "not publicly accessible".
_REPO_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_repo_id(repo_id: str) -> bool:
    """Reject repo IDs that don't look like 'owner/name'.

    Allowing dots means path traversal has to be excluded explicitly: the
    leading-character class stops a segment starting with '.', and this rules
    out '..' anywhere in the id.
    """
    if ".." in repo_id:
        return False
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
