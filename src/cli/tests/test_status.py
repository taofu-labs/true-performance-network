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


def stub_leader(monkeypatch):
    comp = {
        "id": "comp-a", "name": "comp-a", "start_block": 0, "commit_end_block": 10,
        "scoring_end_block": 20, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [{"name": "mmlu", "min_score": 0.5, "weight": 1.0}],
    }
    leader_config_client._cache.clear()
    leader_config_client._cache_time.clear()
    monkeypatch.setattr(
        leader_config_client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [comp]}),
    )


def test_status_falls_back_to_block_zero_when_chain_unreachable(monkeypatch):
    stub_leader(monkeypatch)
    monkeypatch.setattr(chain, "get_subtensor", lambda network=None: (_ for _ in ()).throw(RuntimeError("no chain")))

    result = runner.invoke(app, ["--leader-url", LEADER_URL, "status", "--wallet", "alice"])
    assert result.exit_code == 0
    assert "Current block:" in result.stdout


def test_status_unknown_competition_id_exits_nonzero(monkeypatch):
    stub_leader(monkeypatch)
    monkeypatch.setattr(chain, "get_subtensor", lambda network=None: (_ for _ in ()).throw(RuntimeError("no chain")))

    result = runner.invoke(app, ["--leader-url", LEADER_URL, "status", "--wallet", "alice", "--competition", "missing"])
    assert result.exit_code == 1
    assert "not found" in result.stdout
