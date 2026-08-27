import pytest

from competition import leader_config_client as client


@pytest.fixture(autouse=True)
def clear_cache():
    client._cache.clear()
    client._cache_time.clear()
    yield
    client._cache.clear()
    client._cache_time.clear()


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


def make_spec(comp_id="comp-a", **overrides):
    fields = {
        "id": comp_id,
        "name": "Test",
        "start_block": 0,
        "commit_end_block": 10,
        "scoring_end_block": 20,
        "emission_distribution": [1.0],
        "top_n": 1,
        "benchmarks": [{"name": "mmlu", "min_score": 0.5, "weight": 1.0}],
    }
    fields.update(overrides)
    return fields


def test_get_all_competitions(monkeypatch):
    monkeypatch.setattr(
        client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [make_spec("comp-a")]}),
    )
    specs = client.get_all_competitions("http://leader")
    assert [s.id for s in specs] == ["comp-a"]


def test_get_all_competitions_uses_cache(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append(url)
        return FakeResponse({"competitions": [make_spec("comp-a")]})

    monkeypatch.setattr(client.requests, "get", fake_get)
    client.get_all_competitions("http://leader")
    client.get_all_competitions("http://leader")
    assert len(calls) == 1


def test_get_active_competitions_filters_by_block(monkeypatch):
    monkeypatch.setattr(
        client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [
            make_spec("open", start_block=0, commit_end_block=10, scoring_end_block=20),
            make_spec("far-future", start_block=1000, commit_end_block=1010, scoring_end_block=1020),
        ]}),
    )
    assert len(client.get_active_competitions("http://leader", current_block=5)) == 1
    assert len(client.get_active_competitions("http://leader", current_block=5, force_refresh=True)) == 1


def test_stale_cache_used_when_refetch_fails(monkeypatch):
    monkeypatch.setattr(
        client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [make_spec("comp-a")]}),
    )
    specs = client.get_all_competitions("http://leader")
    assert len(specs) == 1

    def raise_error(url, timeout):
        raise ConnectionError("down")

    monkeypatch.setattr(client.requests, "get", raise_error)
    stale = client.get_all_competitions("http://leader", force_refresh=True)
    assert len(stale) == 1


def test_is_competition(monkeypatch):
    monkeypatch.setattr(
        client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [make_spec("comp-a")]}),
    )
    assert client.is_competition("http://leader", "comp-a") is True
    assert client.is_competition("http://leader", "comp-missing") is False


def test_stale_cache_dropped_past_max_stale(monkeypatch):
    monkeypatch.setattr(
        client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [make_spec("comp-a")]}),
    )
    assert len(client.get_all_competitions("http://leader")) == 1

    def raise_error(url, timeout):
        raise ConnectionError("down")

    monkeypatch.setattr(client.requests, "get", raise_error)
    client._cache_time["http://leader"] -= client._CACHE_MAX_STALE + 1
    assert client.get_all_competitions("http://leader") == []
