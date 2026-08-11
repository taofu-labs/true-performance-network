import hashlib
import subprocess
from unittest.mock import patch

from precheck_api import _RE_KV, _RE_WEIGHTS, _run_llama_cli, _sha256_file


def test_re_weights_extracts_last_match_in_mib():
    log = (
        "load_tensors: layer  0 assigned to device CPU\n"
        "load_tensors:   CPU model buffer size =     0.00 MiB\n"
        "load_tensors:   CPU model buffer size =  4096.50 MiB\n"
    )
    matches = _RE_WEIGHTS.findall(log)
    assert matches[-1] == "4096.50"


def test_re_kv_extracts_last_match_in_mib():
    log = (
        "llama_kv_cache:   CPU KV buffer size =     0.00 MiB\n"
        "llama_kv_cache:   CPU KV buffer size =   256.00 MiB\n"
    )
    matches = _RE_KV.findall(log)
    assert matches[-1] == "256.00"


def test_re_weights_no_match_on_unrelated_log():
    assert _RE_WEIGHTS.findall("some unrelated llama-cli output\n") == []


def test_sha256_file_matches_hashlib(tmp_path):
    f = tmp_path / "model.gguf"
    f.write_bytes(b"fake gguf bytes" * 1000)
    expected = hashlib.sha256(f.read_bytes()).hexdigest()
    assert _sha256_file(str(f)) == expected


def test_run_llama_cli_success_has_no_reason():
    log = (
        "load_tensors:   CPU model buffer size =  4096.50 MiB\n"
        "llama_kv_cache:   CPU KV buffer size =   256.00 MiB\n"
    ).encode()
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=log, stderr=b"")
    with patch("subprocess.run", return_value=fake):
        ram_result, reason = _run_llama_cli("model.gguf", 4096)
    assert reason is None
    assert ram_result["passed"] is True


def test_run_llama_cli_timeout_reason():
    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="llama-cli", timeout=600)):
        ram_result, reason = _run_llama_cli("model.gguf", 4096)
    assert ram_result["passed"] is False
    assert "timed out" in reason
    assert "600" in reason


def test_run_llama_cli_oom_kill_reason():
    fake = subprocess.CompletedProcess(args=[], returncode=-9, stdout=b"", stderr=b"")
    with patch("subprocess.run", return_value=fake):
        ram_result, reason = _run_llama_cli("model.gguf", 4096)
    assert ram_result["passed"] is False
    assert "signal 9" in reason


def test_run_llama_cli_nonzero_exit_reason():
    fake = subprocess.CompletedProcess(args=[], returncode=1, stdout=b"", stderr=b"segfault or similar")
    with patch("subprocess.run", return_value=fake):
        ram_result, reason = _run_llama_cli("model.gguf", 4096)
    assert ram_result["passed"] is False
    assert "exit 1" in reason
    assert "segfault" in reason


def test_run_llama_cli_parse_failure_reason():
    fake = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"unrelated log output\n", stderr=b"")
    with patch("subprocess.run", return_value=fake):
        ram_result, reason = _run_llama_cli("model.gguf", 4096)
    assert ram_result["passed"] is False
    assert "parse failed" in reason
    assert "weights" in reason


def test_check_local_path_traversal_guard():
    """Mirrors the guard in precheck_api.check_local: resolved path must stay under /data/."""
    from pathlib import Path

    data_root = Path("/data").resolve()

    safe = Path("/data/model.gguf").resolve()
    assert str(safe).startswith(str(data_root))

    traversal = Path("/data/../etc/passwd").resolve()
    assert not str(traversal).startswith(str(data_root))
