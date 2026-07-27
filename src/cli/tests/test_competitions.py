from typer.testing import CliRunner

import common.chain as chain
from cli.app import app
from competition import leader_config_client

runner = CliRunner()

LEADER_URL = "http://fake-leader"


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def stub_leader(monkeypatch, extra_fields=None):
    comp = {
        "id": "comp-a", "name": "comp-a", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [{"name": "mmlu", "min_score": 0.5, "weight": 1.0}],
    }
    comp.update(extra_fields or {})
    leader_config_client._cache.clear()
    leader_config_client._cache_time.clear()
    monkeypatch.setattr(
        leader_config_client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [comp]}),
    )


def test_competitions_shows_active_only_by_default(monkeypatch):
    stub_leader(monkeypatch)
    monkeypatch.setattr(chain, "get_subtensor", lambda network=None: (_ for _ in ()).throw(RuntimeError("no chain")))

    result = runner.invoke(app, ["--leader-url", LEADER_URL, "competitions"])
    assert result.exit_code == 0
    assert "comp-a" in result.stdout


def test_competitions_all_flag_shows_inactive_too(monkeypatch):
    stub_leader(monkeypatch, {"start_block": 500, "commit_end_block": 510, "scoring_end_block": 520})
    monkeypatch.setattr(chain, "get_subtensor", lambda network=None: (_ for _ in ()).throw(RuntimeError("no chain")))

    # Not active at block 0, so default (active-only) view finds nothing.
    result = runner.invoke(app, ["--leader-url", LEADER_URL, "competitions"])
    assert "No competitions found" in result.stdout

    result_all = runner.invoke(app, ["--leader-url", LEADER_URL, "competitions", "--all"])
    assert result_all.exit_code == 0
    assert "comp-a" in result_all.stdout
