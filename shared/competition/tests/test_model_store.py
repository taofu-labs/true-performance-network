"""
Repo-id validation.

The regex gates `check_repo_public`, so anything it rejects is reported to the
miner as "repo not publicly accessible" — a misleading message for a valid
repo. It originally excluded dots, which rejected most modern model repos.
"""
import pytest

from competition.model_store import _validate_repo_id, check_repo_public


@pytest.mark.parametrize("repo_id", [
    "user/repo-name_2",
    # Dots are ordinary in real repo names. Excluding them silently failed
    # every miner submitting a current-generation model.
    "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
    "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
])
def test_accepts_real_repo_ids(repo_id):
    assert _validate_repo_id(repo_id) is True


@pytest.mark.parametrize("repo_id", [
    "",
    "user",                 # no owner/name split
    "user/re/po",           # too many segments
    "user/repo extra",      # whitespace
])
def test_rejects_malformed_repo_ids(repo_id):
    assert _validate_repo_id(repo_id) is False


@pytest.mark.parametrize("repo_id", [
    "../evil",
    "user/../etc",
    ".hidden/repo",    # a segment may not start with a dot
    # These two pass the character class — only the explicit ".." check
    # rejects them, so they are what keeps that guard honest.
    "a../b",
    "user/repo..x",
])
def test_rejects_path_traversal(repo_id):
    """Allowing dots must not open a traversal hole — the repo id reaches
    HuggingFace URLs and container-side paths."""
    assert _validate_repo_id(repo_id) is False


def test_check_repo_public_rejects_invalid_id_without_network(monkeypatch):
    """A malformed id must fail closed, before any HTTP request."""
    import competition.model_store as model_store

    def explode(*a, **k):
        raise AssertionError("must not make a request for an invalid repo id")
    monkeypatch.setattr(model_store.requests, "head", explode)

    assert check_repo_public("user/../etc") is False


def test_check_repo_public_network_error_is_not_public(monkeypatch):
    import competition.model_store as model_store

    def boom(*a, **k):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(model_store.requests, "head", boom)

    assert check_repo_public("user/repo") is False
