import subprocess

import pytest
import requests

from competition.precheck_client import PrecheckContainer, _parse_verdict, cleanup_stale_containers


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


def test_check_returns_error_when_not_started():
    ctr = make_container()
    verdict = ctr.check("https://example.com/model.gguf")
    assert verdict.error == "container not running"


def test_check_never_raises_on_connection_error(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("refused")))
    verdict = ctr.check("https://example.com/model.gguf")
    assert verdict.error is not None
    assert "connection error" in verdict.error


def test_check_never_raises_on_timeout(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"
    monkeypatch.setattr(requests, "post", lambda *a, **k: (_ for _ in ()).throw(requests.Timeout()))
    verdict = ctr.check("https://example.com/model.gguf")
    assert verdict.error is not None
    assert "timed out" in verdict.error


def test_check_503_returns_not_ready_error(monkeypatch):
    ctr = make_container()
    ctr._container_name = "fake-container"

    class _Resp:
        status_code = 503
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp())
    verdict = ctr.check("https://example.com/model.gguf")
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


def test_cleanup_stale_containers_removes_leftover_containers(monkeypatch):
    """A killed validator (autoupdater SIGKILLs on every detected update)
    can leave an orphaned tpn-precheck-* container bound to the static host
    port — cleanup on startup must find and remove it so the next start()
    doesn't fail with 'port already allocated'."""
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[:2] == ["docker", "ps"]:
            return _FakeCompletedProcess(returncode=0)._with_stdout("abc123\ndef456\n")
        return _FakeCompletedProcess(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    cleanup_stale_containers()

    assert calls[0][:3] == ["docker", "ps", "-aq"]
    assert calls[1] == ["docker", "rm", "-f", "abc123", "def456"]


def test_cleanup_stale_containers_noop_when_none_found(monkeypatch):
    def fake_run(cmd, **kwargs):
        return _FakeCompletedProcess(returncode=0)._with_stdout("")

    monkeypatch.setattr(subprocess, "run", fake_run)
    cleanup_stale_containers()  # must not raise, must not attempt docker rm


def test_cleanup_stale_containers_swallows_docker_missing(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    cleanup_stale_containers()  # must not raise
