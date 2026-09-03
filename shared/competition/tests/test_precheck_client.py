import subprocess

import pytest
import requests

from competition.precheck_client import (
    PrecheckContainer,
    _parse_verdict,
    cleanup_stale_containers,
    container_port,
    is_container_up,
    stop_container,
)


class _FakeCompletedProcess:
    def __init__(self, returncode=0, stderr="", stdout=""):
        self.returncode = returncode
        self.stderr = stderr
        self.stdout = stdout

    def _with_stdout(self, stdout):
        self.stdout = stdout
        return self


def make_container(**overrides):
    defaults = dict(health_timeout=1, health_poll=0)
    defaults.update(overrides)
    return PrecheckContainer(**defaults)


# ---------------------------------------------------------------------------
# Naming / port derivation
# ---------------------------------------------------------------------------

def test_precheck_container_uses_derived_name_and_port_when_competition_id_given():
    ctr = make_container(competition_id="comp1")
    assert ctr._name == "tpn-precheck-comp1"
    assert ctr._host_port == container_port("comp1")


def test_precheck_container_falls_back_to_random_name_without_competition_id():
    a = make_container()
    b = make_container()
    assert a._name != b._name
    assert a._name.startswith("tpn-precheck-")


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------

def test_start_raises_when_docker_not_found(monkeypatch):
    ctr = make_container()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    with pytest.raises(RuntimeError, match="docker not found"):
        ctr.start()


def test_start_raises_when_docker_run_fails(monkeypatch):
    ctr = make_container()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=1, stderr="boom"))
    with pytest.raises(RuntimeError, match="docker run failed"):
        ctr.start()


def test_start_becomes_ready_when_health_check_passes(monkeypatch):
    ctr = make_container()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=0))

    class _HealthResp:
        ok = True
        def json(self):
            return {"ready": True}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _HealthResp())

    ctr.start()
    assert ctr._container_name is not None


def test_start_is_idempotent_when_already_started(monkeypatch):
    """First start() does one liveness check (docker ps) + one docker run.
    A second start() on the same already-started instance must short-circuit
    immediately — zero additional subprocess calls."""
    ctr = make_container()
    calls = {"n": 0}

    def fake_run(cmd, **kwargs):
        calls["n"] += 1
        return _FakeCompletedProcess(returncode=0)
    monkeypatch.setattr(subprocess, "run", fake_run)

    class _HealthResp:
        ok = True
        def json(self):
            return {"ready": True}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _HealthResp())

    ctr.start()
    after_first = calls["n"]
    ctr.start()  # must not raise, must not touch subprocess again
    assert calls["n"] == after_first


def test_start_reattaches_when_container_already_running_by_name(monkeypatch):
    """A fresh instance for a competition whose container survived a
    validator restart must reattach by name, not try to create a new one."""
    ctr = make_container(competition_id="comp1")

    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return _FakeCompletedProcess(returncode=0)._with_stdout("some-id\n")
        raise AssertionError("must not call docker run when already running")
    monkeypatch.setattr(subprocess, "run", fake_run)

    class _HealthResp:
        ok = True
        def json(self):
            return {"ready": True}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _HealthResp())

    ctr.start()
    assert ctr._container_name == "tpn-precheck-comp1"


# ---------------------------------------------------------------------------
# launch() — non-blocking, never waits for /health readiness
# ---------------------------------------------------------------------------

def test_launch_does_not_wait_for_health(monkeypatch):
    """launch() must return as soon as `docker run -d` returns — never call
    /health at all. Regression guard: run_stage_2 calls launch() every tick
    and would block every sibling competition for hours (base model
    download) if this ever called _wait_until_ready() again."""
    ctr = make_container()
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=0))

    def exploding_get(*a, **k):
        raise AssertionError("launch() must never call /health")
    monkeypatch.setattr(requests, "get", exploding_get)

    ctr.launch()
    assert ctr._container_name is not None


# ---------------------------------------------------------------------------
# check()
# ---------------------------------------------------------------------------

def test_check_returns_error_when_not_started():
    ctr = make_container()
    verdict = ctr.check("user/repo", "a" * 40, "model.gguf")
    assert verdict.error == "container not running"


def test_check_never_raises_on_connection_error(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")))
    verdict = ctr.check("user/repo", "a" * 40, "model.gguf")
    assert verdict.error is not None
    assert "connection error" in verdict.error


def test_check_never_raises_on_timeout(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    verdict = ctr.check("user/repo", "a" * 40, "model.gguf")
    assert verdict.error is not None
    assert "timed out" in verdict.error


def test_check_503_returns_not_ready_error(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"

    class _Resp:
        status_code = 503
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    verdict = ctr.check("user/repo", "a" * 40, "model.gguf")
    assert "not ready" in verdict.error


def test_parse_verdict_full_success_payload():
    data = {
        "provenance": {"is_derivative": True, "cka": 0.9, "notes": []},
        "ram": {"passed": True, "ram_bytes": 100, "weights_bytes": 80, "kv_cache_bytes": 20},
        "sha256": "abc123",
        "error": None,
    }
    verdict = _parse_verdict(data)
    assert verdict.provenance.is_derivative is True
    assert verdict.ram.ram_bytes == 100
    assert verdict.sha256 == "abc123"
    assert verdict.error is None


def test_parse_verdict_missing_optional_sections():
    verdict = _parse_verdict({"provenance": None, "ram": None})
    assert verdict.provenance is None
    assert verdict.ram is None
    assert verdict.sha256 is None


# ---------------------------------------------------------------------------
# Module-level is_container_up / stop_container
# ---------------------------------------------------------------------------

def test_is_container_up_false_when_not_running(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=0)._with_stdout(""))
    assert is_container_up("comp1") is False


def test_is_container_up_true_when_running_and_healthy(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=0)._with_stdout("some-id\n"))

    class _HealthResp:
        ok = True
        def json(self):
            return {"ready": True}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _HealthResp())

    assert is_container_up("comp1") is True


def test_is_container_up_false_when_running_but_not_ready(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: _FakeCompletedProcess(returncode=0)._with_stdout("some-id\n"))

    class _HealthResp:
        ok = True
        def json(self):
            return {"ready": False}
    monkeypatch.setattr(requests, "get", lambda *a, **k: _HealthResp())

    assert is_container_up("comp1") is False


def test_stop_container_calls_docker_rm_by_name(monkeypatch):
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: calls.append(cmd) or _FakeCompletedProcess(returncode=0))
    stop_container("comp1")
    assert calls[0] == ["docker", "rm", "-f", "tpn-precheck-comp1"]


# ---------------------------------------------------------------------------
# cleanup_stale_containers
# ---------------------------------------------------------------------------

def test_cleanup_stale_containers_removes_leftover_containers(monkeypatch):
    """A killed validator can leave an orphaned tpn-precheck-* container —
    cleanup on startup must find and remove any not in keep_competition_ids."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return _FakeCompletedProcess(returncode=0)._with_stdout(
                "abc123 tpn-precheck-stale1\ndef456 tpn-precheck-stale2\n"
            )
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cleanup_stale_containers()

    assert calls[0][:2] == ["docker", "ps"]
    assert set(calls[1]) == {"docker", "rm", "-f", "abc123", "def456"}


def test_cleanup_stale_containers_preserves_kept_competitions(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[:2] == ["docker", "ps"]:
            return _FakeCompletedProcess(returncode=0)._with_stdout(
                "abc123 tpn-precheck-comp1\ndef456 tpn-precheck-comp2\n"
            )
        return _FakeCompletedProcess(returncode=0)

    calls = []
    monkeypatch.setattr(subprocess, "run", lambda cmd, **k: (calls.append(cmd), fake_run(cmd, **k))[1])
    cleanup_stale_containers(keep_competition_ids=["comp1"])

    rm_call = next(c for c in calls if c[:2] == ["docker", "rm"])
    assert rm_call == ["docker", "rm", "-f", "def456"]


def test_cleanup_stale_containers_noop_when_none_found(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0)._with_stdout("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cleanup_stale_containers()  # must not raise, must not attempt docker rm


def test_cleanup_stale_containers_swallows_docker_missing(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    cleanup_stale_containers()  # must not raise
