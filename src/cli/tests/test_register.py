from types import SimpleNamespace

from typer.testing import CliRunner

import cli.commands.register as register_mod
import cli.utils.config as config_mod
from cli.app import app

runner = CliRunner()


def patch_registration_seam(monkeypatch, tmp_path, returncode=0):
    calls = []

    def fake_run(cmd):
        calls.append(cmd)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(register_mod.subprocess, "run", fake_run)
    monkeypatch.setattr(config_mod, "tpn_home", lambda: tmp_path / ".tpn")
    return calls


def test_register_success_creates_identity_dir(monkeypatch, tmp_path):
    calls = patch_registration_seam(monkeypatch, tmp_path)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
    ], input="y\n")

    assert result.exit_code == 0
    assert calls[0][:3] == ["btcli", "subnet", "register"]
    assert (tmp_path / ".tpn" / "alice" / "default").is_dir()


def test_register_failure_exits_nonzero(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path, returncode=1)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
    ], input="y\n")

    assert result.exit_code == 1
    assert not (tmp_path / ".tpn" / "alice").exists()
