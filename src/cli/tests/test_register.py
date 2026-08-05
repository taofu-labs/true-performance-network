from types import SimpleNamespace

from typer.testing import CliRunner

# Pre-import bittensor before any monkeypatching of the `subprocess` module:
# it lazily imports heavy C-extension deps (pycryptodome's cpuid probe via
# platform.architecture()) that shell out through the real `subprocess.run`.
# Patching `subprocess.run` (a process-wide singleton module) before that
# import has already happened breaks the probe with the test's fake stub.
import bittensor  # noqa: F401

import cli.commands.register as register_mod
import cli.utils.config as config_mod
from cli.app import app

runner = CliRunner()


class FakeWallet:
    class hotkey:
        ss58_address = "5FakeHotkey"
    class coldkey:
        ss58_address = "5FakeColdkey"


class FakeSubtensor:
    def __init__(self, shielded_result=None, execute_result=None):
        self.shielded_calls = []
        self.execute_calls = []
        self._shielded_result = shielded_result or SimpleNamespace(success=True, message="ok")
        self._execute_result = execute_result or SimpleNamespace(success=True, message="ok")

    def submit_shielded(self, intent, wallet):
        self.shielded_calls.append(intent)
        return self._shielded_result

    def execute(self, intent, wallet):
        self.execute_calls.append(intent)
        return self._execute_result


def patch_registration_seam(monkeypatch, tmp_path, returncode=0):
    monkeypatch.setattr(register_mod.subprocess, "run", lambda cmd: SimpleNamespace(returncode=returncode))
    monkeypatch.setattr(config_mod, "tpn_home", lambda: tmp_path / ".tpn")


def patch_chain_seam(monkeypatch, subtensor):
    import common.chain as chain_mod
    monkeypatch.setattr(chain_mod, "get_wallet", lambda coldkey, hotkey, wallet_path: FakeWallet())
    monkeypatch.setattr(chain_mod, "get_subtensor", lambda network: subtensor)


def test_register_no_collateral_no_floor_skips_both(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path)
    subtensor = FakeSubtensor()
    patch_chain_seam(monkeypatch, subtensor)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
        "--no-collateral", "--no-floor",
    ], input="y\n")
    assert result.exit_code == 0
    assert subtensor.shielded_calls == []
    assert subtensor.execute_calls == []


def test_register_zero_amounts_skip_via_guard(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path)
    subtensor = FakeSubtensor()
    patch_chain_seam(monkeypatch, subtensor)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
        "--collateral-amount", "0", "--floor-amount", "0",
    ], input="y\n")
    assert result.exit_code == 0
    assert subtensor.shielded_calls == []
    assert subtensor.execute_calls == []


def test_register_submits_collateral_and_floor_on_success(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path)
    subtensor = FakeSubtensor()
    patch_chain_seam(monkeypatch, subtensor)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
        "--collateral-amount", "1.5", "--floor-amount", "0.5",
    ], input="y\n")
    assert result.exit_code == 0
    assert len(subtensor.shielded_calls) == 1
    assert subtensor.shielded_calls[0].amount_alpha.amount == 1.5
    assert len(subtensor.execute_calls) == 1
    assert subtensor.execute_calls[0].min_alpha.amount == 0.5
    assert "Collateral locked" in result.stdout
    assert "Collateral floor set" in result.stdout


def test_register_reports_failed_collateral_without_failing_command(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path)
    subtensor = FakeSubtensor(shielded_result=SimpleNamespace(success=False, message="slippage too high"))
    patch_chain_seam(monkeypatch, subtensor)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
        "--collateral-amount", "1.5", "--no-floor",
    ], input="y\n")
    assert result.exit_code == 0
    assert "Collateral locked failed" in result.stdout
    assert "slippage too high" in result.stdout


def test_register_failure_skips_collateral_entirely(monkeypatch, tmp_path):
    patch_registration_seam(monkeypatch, tmp_path, returncode=1)
    subtensor = FakeSubtensor()
    patch_chain_seam(monkeypatch, subtensor)

    result = runner.invoke(app, [
        "register", "--coldkey", "alice", "--hotkey", "default",
        "--collateral-amount", "1.5", "--floor-amount", "0.5",
    ], input="y\n")
    assert result.exit_code == 1
    assert subtensor.shielded_calls == []
    assert subtensor.execute_calls == []
