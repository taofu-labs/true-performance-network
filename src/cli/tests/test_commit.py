import json

from typer.testing import CliRunner

import cli.commands.commit as commit_mod
import cli.utils.config as config_mod
from cli.app import app
from competition import leader_config_client

runner = CliRunner()

LEADER_URL = "http://fake-leader"


class FakeWallet:
    class hotkey:
        ss58_address = "5FakeHotkey"
    class coldkey:
        ss58_address = "5FakeColdkey"


class FakeSubtensor:
    def block(self):
        return 5


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def stub_leader(monkeypatch):
    comp = {
        "id": "comp-a", "name": "comp-a", "start_block": 0, "commit_end_block": 100,
        "scoring_end_block": 200, "emission_distribution": [1.0], "top_n": 1,
        "benchmarks": [{"name": "mmlu", "min_score": 0.5, "weight": 1.0}],
    }
    leader_config_client._cache.clear()
    leader_config_client._cache_time.clear()
    monkeypatch.setattr(
        leader_config_client.requests, "get",
        lambda url, timeout: FakeResponse({"competitions": [comp]}),
    )


def patch_chain_seam(monkeypatch, tmp_path, registered=True):
    monkeypatch.setattr(commit_mod, "get_wallet", lambda coldkey, hotkey, wallet_path: FakeWallet())
    monkeypatch.setattr(commit_mod, "get_subtensor", lambda network: FakeSubtensor())
    monkeypatch.setattr(commit_mod, "is_hotkey_registered", lambda subtensor, hotkey_ss58, netuid: registered)
    monkeypatch.setattr(config_mod, "tpn_home", lambda: tmp_path / ".tpn")


def test_commit_exits_when_hotkey_not_registered(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=False)

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a", "--dry-run",
    ])
    assert result.exit_code == 1
    assert "not registered" in result.stdout


def test_commit_dry_run_does_not_write_to_chain(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=True)
    monkeypatch.setattr(commit_mod, "timelocked_commit", lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("must not be called on --dry-run")
    ))

    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({
        "repository": "user/repo",
        "file": "model.gguf",
        "file_sha256": "a" * 64,
        "huggingface_revision": "a" * 40,
        "runs": [{"b": "mmlu", "r": "r1234"}],
        "max_memory": 1000,
    }))

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a",
        "--config", str(config_file), "--dry-run",
    ])
    assert result.exit_code == 0
    assert "Dry run" in result.stdout


def test_commit_missing_upload_fields_exits_nonzero(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=True)

    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({}))

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a",
        "--config", str(config_file), "--dry-run",
    ])
    assert result.exit_code == 1
    assert "Missing upload data" in result.stdout


def test_commit_accepts_runs_flag(monkeypatch, tmp_path):
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=True)

    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({
        "repository": "user/repo",
        "file": "model.gguf",
        "file_sha256": "a" * 64,
        "huggingface_revision": "a" * 40,
        "max_memory": 1000,
    }))

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a",
        "--config", str(config_file), "--runs", '[{"b":"mmlu","r":"r1234"}]',
        "--dry-run",
    ])
    assert result.exit_code == 0
    assert "r1234" in result.stdout


def test_commit_rejects_malformed_run_id_before_writing_to_chain(monkeypatch, tmp_path):
    """A typo must fail at commit time. Left to reveal, it would silently
    score 0.0 for that benchmark with no way to correct it."""
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=True)

    config_file = tmp_path / "cfg.json"
    config_file.write_text(json.dumps({
        "repository": "user/repo",
        "file": "model.gguf",
        "file_sha256": "a" * 64,
        "huggingface_revision": "a" * 40,
        "max_memory": 1000,
    }))

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a",
        "--config", str(config_file), "--runs", '[{"b":"mmlu","r":"not a run id"}]',
        "--dry-run",
    ])
    assert result.exit_code == 1
    assert "Invalid run id" in result.stdout



def test_commit_saves_runs_to_config(monkeypatch, tmp_path):
    """`tpn status` and a later re-commit both read runs back out of here."""
    stub_leader(monkeypatch)
    patch_chain_seam(monkeypatch, tmp_path, registered=True)

    import cli.utils.config as cfg_mod
    cfg_mod.save_competition_config("alice", "default", "comp-a", {
        "repository": "user/repo", "file": "model.gguf", "file_sha256": "a" * 64,
        "huggingface_revision": "a" * 40, "max_memory": 1000,
    })

    result = runner.invoke(app, [
        "--leader-url", LEADER_URL,
        "commit", "--wallet", "alice", "--competition", "comp-a",
        "--runs", '[{"b":"mmlu","r":"r1234"}]', "--dry-run",
    ])
    assert result.exit_code == 0, result.stdout

    saved = cfg_mod.load_competition_config("alice", "default", "comp-a")
    assert saved["runs"] == [{"b": "mmlu", "r": "r1234"}]
