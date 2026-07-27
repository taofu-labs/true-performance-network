import pytest

from competition.benchmark_client import (
    HttpCoordinator,
    MockCoordinator,
    RunStatusCode,
    _extract_scores,
    make_coordinator,
)


def test_mock_coordinator_submit_is_deterministic():
    c = MockCoordinator()
    run_id_a = c.submit("user/model", "a" * 40, "mmlu")
    run_id_b = c.submit("user/model", "a" * 40, "mmlu")
    assert run_id_a == run_id_b


def test_mock_coordinator_poll_transitions_running_then_completed():
    c = MockCoordinator()
    run_id = c.submit("user/model", "a" * 40, "mmlu")
    statuses = [c.poll(run_id).status for _ in range(3)]
    assert statuses[0] == RunStatusCode.RUNNING
    assert statuses[-1] == RunStatusCode.COMPLETED


def test_mock_coordinator_poll_unknown_run_id_fails():
    c = MockCoordinator()
    status = c.poll("00000000-0000-0000-0000-000000000000")
    assert status.status == RunStatusCode.FAILED


def test_mock_coordinator_list_benchmarks_contains_expected():
    assert "mmlu" in MockCoordinator().list_benchmarks()


def test_extract_scores_lm_eval_nested_shape():
    result = {"results": {"mmlu": {"acc,none": 0.71}}}
    assert _extract_scores(result, "mmlu") == {"mmlu": 0.71}


def test_extract_scores_result_summary_maps_score_to_benchmark_name():
    result = {"score": 0.42, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "samples": 20}
    assert _extract_scores(result, "gsm8k") == {"gsm8k": 0.42}


def test_extract_scores_flat_dict_keyed_by_benchmark():
    assert _extract_scores({"mmlu": 0.7, "gsm8k": 0.3}, "mmlu") == {"mmlu": 0.7, "gsm8k": 0.3}


def test_extract_scores_empty_result_returns_empty():
    assert _extract_scores(None, "mmlu") == {}
    assert _extract_scores({}, "mmlu") == {}


def test_make_coordinator_defaults_to_mock(monkeypatch):
    monkeypatch.delenv("BENCHMARK_BACKEND", raising=False)
    assert isinstance(make_coordinator(), MockCoordinator)


def test_make_coordinator_http_requires_api_key(monkeypatch):
    monkeypatch.setenv("BENCHMARK_BACKEND", "http")
    monkeypatch.delenv("COORDINATOR_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_coordinator()


def test_make_coordinator_http_with_api_key(monkeypatch):
    monkeypatch.setenv("BENCHMARK_BACKEND", "http")
    monkeypatch.setenv("COORDINATOR_API_KEY", "test-key")
    coordinator = make_coordinator()
    assert isinstance(coordinator, HttpCoordinator)


class _FakeResp:
    def __init__(self, payload, ok=True):
        self._payload = payload
        self.ok = ok

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_http_coordinator_poll_cache_hit_is_completed():
    http = HttpCoordinator(base_url="http://example.invalid", api_key="k")
    http._run_benchmark["run1"] = "gsm8k"
    http._session.get = lambda *a, **k: _FakeResp({"status": "cache_hit", "result": {"gsm8k": 0.9}})
    status = http.poll("run1")
    assert status.status == RunStatusCode.COMPLETED
    assert status.scores == {"gsm8k": 0.9}


def test_http_coordinator_poll_cancelled_is_failed():
    http = HttpCoordinator(base_url="http://example.invalid", api_key="k")
    http._session.get = lambda *a, **k: _FakeResp({"status": "cancelled"})
    status = http.poll("run1")
    assert status.status == RunStatusCode.FAILED


def test_http_coordinator_poll_in_progress_status_is_running():
    http = HttpCoordinator(base_url="http://example.invalid", api_key="k")
    http._session.get = lambda *a, **k: _FakeResp({"status": "benchmarking"})
    status = http.poll("run1")
    assert status.status == RunStatusCode.RUNNING
