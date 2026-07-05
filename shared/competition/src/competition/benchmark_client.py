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
from loguru import logger


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

    def list_benchmarks(self) -> set:
        if self._benchmark_cache is not None:
            return self._benchmark_cache
        resp = self._session.get(f"{self._base}/benchmarks", timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        self._benchmark_cache = {b["benchmark"] for b in data.get("benchmarks", [])}
        return self._benchmark_cache

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

        # Coordinator may return a cached/completed result immediately
        if data.get("cached") and data.get("result"):
            logger.info(f"[http] cached result for {repo}@{revision[:8]} benchmark={benchmark}")
            self._completed_cache[run_id] = _extract_scores(data["result"], benchmark)

        logger.debug(f"[http] submitted {benchmark} for {repo}@{revision[:8]} -> {run_id}")
        return run_id

    def poll(self, run_id: str) -> RunStatus:
        if run_id in self._completed_cache:
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores=self._completed_cache[run_id])
        resp = self._session.get(f"{self._base}/status/{run_id}", timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()

        raw_status = data.get("status", "")

        if raw_status == "completed":
            scores = _extract_scores(data.get("result") or data.get("result_summary") or {}, None)
            return RunStatus(run_id=run_id, status=RunStatusCode.COMPLETED, scores=scores)

        if raw_status == "failed":
            return RunStatus(
                run_id=run_id,
                status=RunStatusCode.FAILED,
                failure_reason=data.get("failure_reason") or "unknown",
            )

        if raw_status in _IN_PROGRESS:
            return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING)

        # Unknown status — treat as still running
        logger.warning(f"[http] unknown status {raw_status!r} for {run_id}")
        return RunStatus(run_id=run_id, status=RunStatusCode.RUNNING)


def _extract_scores(result: dict, benchmark: Optional[str]) -> Dict[str, float]:
    """
    Coordinator result shape varies by engine/benchmark. Try common keys:
    {"results": {"mmlu": {"acc,none": 0.7}}} (lm_eval)
    {"mmlu": 0.7}
    {"score": 0.7}  with benchmark name as key
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

    print(f"OK — mmlu score={status.scores['mmlu']:.4f}, run_id={run_id}")
