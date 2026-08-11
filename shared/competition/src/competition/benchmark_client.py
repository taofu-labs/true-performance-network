"""
Coordinator client for the LLM Benchmark Coordinator API.

Usage:
    from competition.benchmark_client import make_coordinator

    coordinator = make_coordinator()
    available  = coordinator.list_benchmarks()          # set[str]
    run_id     = coordinator.submit(repo, revision, benchmark, model_files, quantization)
    status     = coordinator.poll(run_id)               # RunStatus
    scores     = status.scores                          # Dict[str, float] when completed

Select backend via env:
    BENCHMARK_BACKEND=mock   (default) — deterministic fake runs, no network
    BENCHMARK_BACKEND=http   — real HTTP calls to COORDINATOR_BASE_URL
"""
import os
import time
import uuid
import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import requests
import tenacity
from loguru import logger

# ---------------------------------------------------------------------------
# Retry policy — coordinator backpressure (429) and transient 5xx/network
# errors should be retried, not treated as participant failure. Terminal
# client errors (auth, malformed request, unknown benchmark, invalid model
# payload — any other 4xx) are NOT retried.
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

SUBMIT_RETRY_COUNT = int(os.getenv("REQUEST_RETRY_COUNT", 3))
SUBMIT_RETRY_BASE_DELAY = float(os.getenv("COORDINATOR_RETRY_BASE_DELAY", 2.0))
SUBMIT_RETRY_MAX_DELAY = float(os.getenv("COORDINATOR_RETRY_MAX_DELAY", 60.0))


def _is_retryable_error(exc: BaseException) -> bool:
    if isinstance(exc, requests.HTTPError):
        resp = exc.response
        return resp is not None and resp.status_code in _RETRYABLE_STATUS_CODES
    return isinstance(exc, (requests.Timeout, requests.ConnectionError))


def _retry_after_wait(retry_state: "tenacity.RetryCallState") -> float:
    """Honor a Retry-After header on 429s; otherwise exponential backoff."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    if isinstance(exc, requests.HTTPError) and exc.response is not None:
        retry_after = exc.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return min(float(retry_after), SUBMIT_RETRY_MAX_DELAY)
            except ValueError:
                pass
    return min(
        SUBMIT_RETRY_BASE_DELAY * (2 ** (retry_state.attempt_number - 1)),
        SUBMIT_RETRY_MAX_DELAY,
    )


def _log_coordinator_retry(retry_state: "tenacity.RetryCallState") -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(f"Coordinator request retry attempt {retry_state.attempt_number} after: {exc}")


def coordinator_retry(func):
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(SUBMIT_RETRY_COUNT),
        wait=_retry_after_wait,
        retry=tenacity.retry_if_exception(_is_retryable_error),
        before_sleep=_log_coordinator_retry,
        reraise=True,
    )(func)


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class RunStatusCode:
    QUEUED      = "queued"
    RUNNING     = "running"   # any in-progress phase
    COMPLETED   = "completed"
    FAILED      = "failed"

# Coordinator in-progress statuses (not terminal)
_IN_PROGRESS = {
    "requested", "quoted", "queued", "provisioning",
    "worker_booting", "downloading_model", "hashing_model",
    "preflight", "benchmarking", "collecting_results",
}


@dataclass
class RunStatus:
    run_id: str
    status: str           # RunStatusCode constant
    scores: Dict[str, float] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    phase: Optional[str] = None
    percent_complete: Optional[float] = None
    last_log_at: Optional[str] = None
    message: Optional[str] = None
    estimated_seconds_remaining: Optional[float] = None


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Coordinator(Protocol):
    def list_benchmarks(self) -> set:
        """Return set of supported benchmark names."""
        ...

    def submit(
        self,
        repo: str,
        revision: str,
        benchmark: str,
        model_files: Optional[List[str]] = None,
        quantization: Optional[str] = None,
    ) -> str:
        """Submit a benchmark run. Returns run_id."""
        ...

    def poll(self, run_id: str) -> RunStatus:
        """Return current run status."""
        ...


# ---------------------------------------------------------------------------
# Mock implementation (default)
# ---------------------------------------------------------------------------

# Public benchmarks from API spec
_PUBLIC_BENCHMARKS = {"mmlu", "gsm8k", "hellaswag", "truthfulqa", "arc_challenge"}

# Completed after this many poll calls (simulates ~2 min benchmark)
_MOCK_POLLS_TO_COMPLETE = 3


class MockCoordinator:
    """Deterministic fake coordinator. No network. Scores derived from run_id hash."""

    def __init__(self):
        # run_id -> poll count
        self._polls: Dict[str, int] = {}
        # run_id -> benchmark name
        self._benchmarks: Dict[str, str] = {}

    def list_benchmarks(self) -> set:
        return set(_PUBLIC_BENCHMARKS)

    def submit(
        self,
        repo: str,
        revision: str,
        benchmark: str,
        model_files: Optional[List[str]] = None,
        quantization: Optional[str] = None,
    ) -> str:
        # Deterministic run_id: same repo+rev+benchmark always returns same id
        seed = f"{repo}:{revision}:{benchmark}"
        run_id = str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))
        self._polls[run_id] = 0
        self._benchmarks[run_id] = benchmark
        logger.debug(f"[mock] submitted {benchmark} for {repo}@{revision[:8]} -> {run_id}")
        return run_id

    def poll(self, run_id: str) -> RunStatus:
        if run_id not in self._polls:
            return RunStatus(run_id=run_id, status=RunStatusCode.FAILED, failure_reason="unknown run_id")

        self._polls[run_id] += 1
        count = self._polls[run_id]

        if count < _MOCK_POLLS_TO_COMPLETE:
            logger.debug(f"[mock] {run_id[:8]} poll {count}/{_MOCK_POLLS_TO_COMPLETE} -> running")
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING)

        # Deterministic score in [0.55, 0.85] from run_id hash
        h = int(hashlib.sha256(run_id.encode()).hexdigest(), 16)
        score = 0.55 + (h % 10000) / 10000 * 0.30
        benchmark = self._benchmarks[run_id]
        logger.debug(f"[mock] {run_id[:8]} completed {benchmark}={score:.4f}")
        return RunStatus(
            run_id=run_id,
            status=RunStatusCode.COMPLETED,
            scores={benchmark: round(score, 4)},
        )


# ---------------------------------------------------------------------------
# HTTP implementation
# ---------------------------------------------------------------------------

class HttpCoordinator:
    """Real HTTP calls to the coordinator API."""

    def __init__(self, base_url: str, api_key: str, timeout: int = 30):
        self._base = base_url.rstrip("/")
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })
        self._timeout = timeout
        self._benchmark_cache: Optional[set] = None
        self._completed_cache: Dict[str, Dict[str, float]] = {}
        self._run_benchmark: Dict[str, str] = {}

    def list_benchmarks(self) -> set:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        resp = self._session.get(f"{self._base}/benchmarks", timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        self._benchmark_cache = {b["benchmark"] for b in data.get("benchmarks", [])}
        return self._benchmark_cache

    @coordinator_retry
    def submit(
        self,
        repo: str,
        revision: str,
        benchmark: str,
        model_files: Optional[List[str]] = None,
        quantization: Optional[str] = None,
    ) -> str:
        body: dict = {
            "huggingface_repo": repo,
            "huggingface_revision": revision,
            "benchmark": benchmark,
            "model_format": "gguf",
        }
        if model_files:
            body["model_files"] = model_files
        if quantization:
            body["quantization"] = quantization

        resp = self._session.post(f"{self._base}/benchmark", json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        run_id = data["run_id"]
        self._run_benchmark[run_id] = benchmark

        # Coordinator may return a cached/completed result immediately
        if data.get("cached") and data.get("result"):
            logger.info(f"[http] cached result for {repo}@{revision[:8]} benchmark={benchmark}")
            self._completed_cache[run_id] = _extract_scores(data["result"], benchmark)

        logger.debug(f"[http] submitted {benchmark} for {repo}@{revision[:8]} -> {run_id}")
        return run_id

    @coordinator_retry
    def poll(self, run_id: str) -> RunStatus:
        if run_id in self._completed_cache:
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores=self._completed_cache[run_id])
        resp = self._session.get(f"{self._base}/status/{run_id}", timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        raw_status = data.get("status", "")
        # Coordinator's status response is the source of truth for which
        # benchmark a run_id belongs to — falls back to the in-memory cache
        # (populated by submit() in this process) for older coordinators that
        # don't echo `benchmark` back, or as a fast path.
        benchmark = data.get("benchmark") or self._run_benchmark.get(run_id)
        progress = data.get("progress") or {}
        progress_fields = dict(
            phase=data.get("phase"),
            percent_complete=data.get("percent_complete"),
            last_log_at=progress.get("last_log_at"),
            message=progress.get("message"),
            estimated_seconds_remaining=progress.get("estimated_seconds_remaining"),
        )

        if raw_status in ("completed", "cache_hit"):
            scores = _extract_scores(data.get("result"), benchmark) or _extract_scores(data.get("result_summary"), benchmark)
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores=scores, **progress_fields)

        if raw_status in ("failed", "cancelled", "cleanup_failed"):
            return RunStatus(
                run_id=run_id,
                status=RunStatusCode.FAILED,
                failure_reason=data.get("failure_reason") or f"run ended in terminal status {raw_status!r}",
                **progress_fields,
            )

        if raw_status in _IN_PROGRESS:
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, **progress_fields)

        # Unknown status — treat as still running
        logger.warning(f"[http] unknown status {raw_status!r} for {run_id}")
        return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING, **progress_fields)


def _extract_scores(result: Optional[dict], benchmark: Optional[str]) -> Dict[str, float]:
    """
    Coordinator result shape varies by engine/benchmark and by which field was
    populated (`result` vs. `/status`'s single-benchmark `result_summary` fallback).
    Try, in order:
    {"results": {"mmlu": {"acc,none": 0.7}}}  (lm_eval `result`)
    {"mmlu": 0.7}                              (flat `result`, keyed by benchmark name)
    {"score": 0.7, "prompt_tokens": ..., ...}  (`result_summary` — single benchmark,
                                                 `score` must be mapped to `benchmark`
                                                 explicitly, not grabbed as a flat dict —
                                                 the other numeric keys are token/sample
                                                 counts, not scores)
    """
    if not result:
        return {}

    # lm_eval nested: result["results"][task]["acc,none"] or ["acc"]
    if "results" in result and isinstance(result["results"], dict):
        scores = {}
        for task, metrics in result["results"].items():
            if isinstance(metrics, dict):
                score = metrics.get("acc,none") or metrics.get("acc") or metrics.get("score")
                if score is not None:
                    # Strip lm_eval suffixes to get plain benchmark name
                    plain = task.split(",")[0].split("|")[0]
                    scores[plain] = float(score)
        if scores:
            return scores

    # result_summary shape: single benchmark's score, plus token/sample counts
    # that must NOT be mistaken for per-benchmark scores.
    if "score" in result and any(k in result for k in ("metrics", "prompt_tokens", "completion_tokens", "total_tokens", "samples")):
        if benchmark and result.get("score") is not None:
            return {benchmark: float(result["score"])}
        return {}

    # Flat dict keyed by benchmark name
    flat = {k: float(v) for k, v in result.items() if isinstance(v, (int, float))}
    if flat:
        return flat

    return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_coordinator() -> Coordinator:
    backend = os.getenv("BENCHMARK_BACKEND", "mock").lower()
    if backend == "http":
        base_url = os.getenv("COORDINATOR_BASE_URL", "https://bench.trueperformancenetwork.com")
        api_key = os.getenv("COORDINATOR_API_KEY", "")
        if not api_key:
            raise RuntimeError("COORDINATOR_API_KEY required for BENCHMARK_BACKEND=http")
        logger.info(f"Coordinator: HTTP -> {base_url}")
        return HttpCoordinator(base_url=base_url, api_key=api_key)

    logger.info("Coordinator: mock (set BENCHMARK_BACKEND=http for real runs)")
    return MockCoordinator()


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    c = MockCoordinator()

    benchmarks = c.list_benchmarks()
    assert "mmlu" in benchmarks, f"mmlu missing: {benchmarks}"
    assert "hellaswag" in benchmarks

    run_id = c.submit("user/model", "abc123def456" + "0" * 28, "mmlu")
    assert run_id, "no run_id"

    # Same inputs -> same run_id (deterministic)
    run_id2 = c.submit("user/model", "abc123def456" + "0" * 28, "mmlu")
    assert run_id == run_id2, "non-deterministic run_id"

    # Poll until completed
    for i in range(_MOCK_POLLS_TO_COMPLETE + 1):
        status = c.poll(run_id)
        if status.status == RunStatusCode.COMPLETED:
            break
    assert status.status == RunStatusCode.COMPLETED, f"never completed: {status}"
    assert "mmlu" in status.scores, f"no scores: {status}"
    assert 0.55 <= status.scores["mmlu"] <= 0.85, f"score out of range: {status.scores}"

    # Unknown run_id -> FAILED
    bad = c.poll("00000000-0000-0000-0000-000000000000")
    assert bad.status == RunStatusCode.FAILED

    # _extract_scores: result_summary shape must map `score` to the polled
    # benchmark name, not scoop up token/sample counts as extra "scores".
    summary_scores = _extract_scores(
        {"score": 0.42, "metrics": None, "prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150, "samples": 20},
        "gsm8k",
    )
    assert summary_scores == {"gsm8k": 0.42}, f"result_summary extraction wrong: {summary_scores}"

    # _extract_scores: lm_eval nested result still takes priority over any
    # result_summary-shaped keys.
    nested_scores = _extract_scores({"results": {"mmlu": {"acc,none": 0.71}}}, "mmlu")
    assert nested_scores == {"mmlu": 0.71}, f"nested extraction wrong: {nested_scores}"

    # HttpCoordinator.poll: cache_hit and cancelled/cleanup_failed terminal
    # statuses must not be treated as still-running.
    class _FakeResp:
        def __init__(self, payload):
            self._payload = payload
        def raise_for_status(self):
            pass
        def json(self):
            return self._payload

    http = HttpCoordinator(base_url="http://example.invalid", api_key="k")
    http._run_benchmark["run-cache-hit"] = "gsm8k"
    http._session.get = lambda *a, **k: _FakeResp({"status": "cache_hit", "result": {"gsm8k": 0.9}})
    cache_hit_status = http.poll("run-cache-hit")
    assert cache_hit_status.status == RunStatusCode.COMPLETED, f"cache_hit not treated as completed: {cache_hit_status}"
    assert cache_hit_status.scores == {"gsm8k": 0.9}, f"cache_hit scores wrong: {cache_hit_status.scores}"

    for terminal_status in ("cancelled", "cleanup_failed"):
        http._session.get = lambda *a, _s=terminal_status, **k: _FakeResp({"status": _s})
        terminal = http.poll("run-terminal")
        assert terminal.status == RunStatusCode.FAILED, f"{terminal_status} not treated as failed: {terminal}"

    print(f"OK — mmlu score={status.scores['mmlu']:.4f}, run_id={run_id}")
