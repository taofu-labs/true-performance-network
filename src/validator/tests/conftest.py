import pytest

import validator.scorer as scorer


@pytest.fixture(autouse=True)
def repos_are_public(monkeypatch):
    monkeypatch.setattr(scorer, "check_repo_public", lambda repo: True)
