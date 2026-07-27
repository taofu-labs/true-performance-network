"""
Validator-side manager for the dockerised precheck API.

The precheck container runs two checks on each submitted GGUF (one download each):
  1. Provenance (optional) — CKA embedding check against base model.
  2. RAM measurement — llama-cli CPU-only load; extracts weights + KV cache bytes.

Lifecycle (per scoring run):
    ctr = PrecheckContainer(base_repo=spec.model_repo)  # None → RAM-only mode
    ctr.start()                                          # blocks until /health ready
    verdict = ctr.check(url, context_length=4096)
    ctr.stop()

Or use as context manager:
    with PrecheckContainer(base_repo) as ctr:
        verdict = ctr.check(url)

Config (env):
    PRECHECK_IMAGE          Docker image tag  (default: ghcr.io/taofulabs/tpn-precheck; set to
                            a bare local tag like "tpn-precheck" to use a locally built image)
    PRECHECK_HOST_PORT      Host port to bind  (default: 8081)
    PRECHECK_HEALTH_TIMEOUT Wall-clock seconds to wait for base model ready (default: 7200)
    PRECHECK_HEALTH_POLL    Seconds between /health polls (default: 15)
    PRECHECK_CHECK_TIMEOUT  HTTP timeout for /check calls (default: 900)
    BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS  Forwarded to container (default: 7200)
"""
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from loguru import logger

from common.settings import LOG_LEVEL

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_IMAGE          = os.environ.get("PRECHECK_IMAGE", "ghcr.io/taofulabs/tpn-precheck")
_HOST_PORT      = int(os.environ.get("PRECHECK_HOST_PORT", "8081"))
_HEALTH_TIMEOUT = int(os.environ.get("PRECHECK_HEALTH_TIMEOUT", "7200"))
_HEALTH_POLL    = int(os.environ.get("PRECHECK_HEALTH_POLL", "15"))
_CHECK_TIMEOUT  = int(os.environ.get("PRECHECK_CHECK_TIMEOUT", "900"))
_BASE_DL_TIMEOUT = os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceResult:
    is_derivative: bool
    cka: float = 0.0
    notes: List[str] = field(default_factory=list)


@dataclass
class RamResult:
    passed: bool
    ram_bytes: int = 0
    weights_bytes: int = 0
    kv_cache_bytes: int = 0


@dataclass
class PrecheckVerdict:
    provenance: Optional[ProvenanceResult]  # None if no base model / not run
    ram: Optional[RamResult]               # None if container error before llama run
    sha256: Optional[str] = None           # None if hash failed or not computed
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Container manager
# ---------------------------------------------------------------------------

class PrecheckContainer:
    """
    Manages one Docker container running precheck_api.py.

    Thread-safe for sequential use within a single scoring run.
    Not safe for concurrent calls to check() — one check at a time.
    """

    def __init__(
        self,
        base_repo: Optional[str] = None,
        image: str = _IMAGE,
        host_port: int = _HOST_PORT,
        health_timeout: int = _HEALTH_TIMEOUT,
        health_poll: int = _HEALTH_POLL,
        check_timeout: int = _CHECK_TIMEOUT,
    ):
        self._base_repo = base_repo or ""
        self._image = image
        self._host_port = host_port
        self._health_timeout = health_timeout
        self._health_poll = health_poll
        self._check_timeout = check_timeout
        self._container_name: Optional[str] = None
        self._base_url: str = f"http://localhost:{host_port}"

    def __enter__(self) -> "PrecheckContainer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        if self._container_name:
            raise RuntimeError("PrecheckContainer already started")

        name = f"tpn-precheck-{uuid.uuid4().hex[:8]}"
        cmd = ["docker", "run", "-d", "--name", name, "-p", f"{self._host_port}:8080", "--rm",
               "-e", f"PRECHECK_LOG_LEVEL={LOG_LEVEL}"]
        if self._base_repo:
            cmd += ["-e", f"BASE_MODEL_REPO={self._base_repo}",
                    "-e", f"BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS={_BASE_DL_TIMEOUT}"]
        cmd.append(self._image)

        logger.info(f"Starting precheck container {name} (base_repo={self._base_repo or 'none'})")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise RuntimeError("docker run timed out after 30s")
        except FileNotFoundError:
            raise RuntimeError("docker not found — install Docker")

        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        self._container_name = name
        logger.info(
            f"Container {name} started — "
            f"{'waiting for base model download' if self._base_repo else 'ready immediately'} "
            f"(up to {self._health_timeout}s)"
        )
        self._wait_until_ready()

    def stop(self) -> None:
        if not self._container_name:
            return
        name = self._container_name
        self._container_name = None
        logger.info(f"Stopping precheck container {name}")
        try:
            subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
        except Exception as e:
            logger.warning(f"Failed to remove container {name}: {e}")

    # ── Check ──────────────────────────────────────────────────────────

    def check(self, gguf_download_url: str, context_length: int = 4096) -> PrecheckVerdict:
        """
        POST download URL to container; runs provenance + RAM check; deletes GGUF.
        Never raises — returns PrecheckVerdict with error set on failure.
        """
        if not self._container_name:
            return PrecheckVerdict(provenance=None, ram=None, error="container not running")

        try:
            resp = requests.post(
                f"{self._base_url}/check",
                json={"url": gguf_download_url, "context_length": context_length},
                timeout=self._check_timeout,
            )
        except requests.Timeout:
            return PrecheckVerdict(
                provenance=None, ram=None,
                error=f"/check timed out after {self._check_timeout}s",
            )
        except requests.ConnectionError as e:
            return PrecheckVerdict(provenance=None, ram=None, error=f"connection error: {e}")

        if resp.status_code == 503:
            return PrecheckVerdict(provenance=None, ram=None, error="container not ready (503)")

        try:
            data = resp.json()
        except Exception:
            return PrecheckVerdict(
                provenance=None, ram=None,
                error=f"non-JSON response {resp.status_code}: {resp.text[:200]}",
            )

        if not resp.ok:
            detail = data.get("detail") or data.get("error") or resp.text[:200]
            return PrecheckVerdict(provenance=None, ram=None, error=f"API error {resp.status_code}: {detail}")

        return _parse_verdict(data)

    # ── Internal ───────────────────────────────────────────────────────

    def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + self._health_timeout
        last_log = 0.0

        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{self._base_url}/health", timeout=5)
                if resp.ok:
                    data = resp.json()
                    if data.get("ready"):
                        logger.info("Precheck container ready")
                        return
                    if data.get("error"):
                        raise RuntimeError(f"Precheck container startup failed: {data['error']}")
            except requests.ConnectionError:
                pass
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(f"Health poll error (will retry): {e}")

            now = time.monotonic()
            if now - last_log >= 60:
                elapsed = int(now - (deadline - self._health_timeout))
                logger.info(f"Waiting for precheck container... {elapsed}s elapsed")
                last_log = now

            time.sleep(self._health_poll)

        raise RuntimeError(
            f"Precheck container not ready after {self._health_timeout}s"
        )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_verdict(data: dict) -> PrecheckVerdict:
    prov_data = data.get("provenance")
    ram_data = data.get("ram")

    provenance = None
    if prov_data and isinstance(prov_data, dict):
        provenance = ProvenanceResult(
            is_derivative=bool(prov_data.get("is_derivative", False)),
            cka=float(prov_data.get("cka", 0.0)),
            notes=list(prov_data.get("notes", [])),
        )

    ram = None
    if ram_data and isinstance(ram_data, dict):
        ram = RamResult(
            passed=bool(ram_data.get("passed", False)),
            ram_bytes=int(ram_data.get("ram_bytes", 0)),
            weights_bytes=int(ram_data.get("weights_bytes", 0)),
            kv_cache_bytes=int(ram_data.get("kv_cache_bytes", 0)),
        )

    return PrecheckVerdict(
        provenance=provenance,
        ram=ram,
        sha256=data.get("sha256") or None,
        error=data.get("error") or None,
    )
