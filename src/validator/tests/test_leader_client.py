import pytest
import requests

from validator.leader_client import LeaderClient


def test_requires_base_url():
    with pytest.raises(ValueError):
        LeaderClient("")


def test_get_scoring_results_returns_results_and_status(monkeypatch):
    client = LeaderClient("http://leader.local:9200")

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {"results": [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "max_memory_kb": 1000}], "scored_status": "scored"}

    captured = {}
    def fake_get(url, params, timeout):
        captured["url"] = url
        captured["params"] = params
        return _Resp()
    monkeypatch.setattr(requests, "get", fake_get)

    results, scored_status = client.get_scoring_results("comp1")
    assert results == [{"hotkey": "hk1", "competition_id": "comp1", "final_score": 0.7, "max_memory_kb": 1000}]
    assert scored_status == "scored"
    assert captured["url"] == "http://leader.local:9200/v1/follower/scoring-results"
    assert captured["params"] == {"competition_id": "comp1"}


def test_get_scoring_results_empty_when_no_results_key(monkeypatch):
    client = LeaderClient("http://leader.local:9200")

    class _Resp:
        def raise_for_status(self):
            pass
        def json(self):
            return {}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())

    assert client.get_scoring_results("comp1") == ([], None)


def test_get_scoring_results_raises_on_http_error(monkeypatch):
    client = LeaderClient("http://leader.local:9200")

    class _Resp:
        def raise_for_status(self):
            raise requests.HTTPError("500")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp())

    with pytest.raises(requests.HTTPError):
        client.get_scoring_results("comp1")
