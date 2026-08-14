"""
Validator-side manager for the dockerised precheck API.

The precheck container runs two checks on each submitted GGUF (one download each):
  1. Provenance (optional) — CKA embedding check against base model.
  2. RAM measurement — llama-cli CPU-only load; extracts weights + KV cache bytes.

Lifecycle (per scoring run):
    ctr = PrecheckContainer(competition_id="comp1", base_repo=spec.model_repo)
    ctr.start()                                          # blocks until /health ready
    verdict = ctr.check(url, context_length=4096)
    ctr.stop()

Or use as context manager:
    with PrecheckContainer(competition_id="comp1", base_repo=spec.model_repo) as ctr:
        verdict = ctr.check(url)

Non-blocking equivalents: launch() (start without waiting), is_container_up(),
stop_container(competition_id).

Config (env):
    PRECHECK_IMAGE          Docker image tag  (default: ghcr.io/taofu-labs/tpn-precheck; set to
                            a bare local tag like "tpn-precheck" to use a locally built image)
    PRECHECK_HOST_PORT_BASE Base of the per-competition host port range (default: 8081)
    PRECHECK_HOST_PORT_RANGE Size of the port range competition ports hash into (default: 200)
    PRECHECK_HEALTH_TIMEOUT Wall-clock seconds to wait for base model ready (default: 7200)
    PRECHECK_HEALTH_POLL    Seconds between /health polls (default: 15)
    PRECHECK_CHECK_TIMEOUT  HTTP timeout for /check calls (default: 900)
    BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS  Forwarded to container (default: 7200)
"""
import hashlib
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

_IMAGE            = os.environ.get("PRECHECK_IMAGE", "ghcr.io/taofu-labs/tpn-precheck")
_HOST_PORT_BASE   = int(os.environ.get("PRECHECK_HOST_PORT_BASE", "8081"))
_HOST_PORT_RANGE  = int(os.environ.get("PRECHECK_HOST_PORT_RANGE", "200"))
_HEALTH_TIMEOUT   = int(os.environ.get("PRECHECK_HEALTH_TIMEOUT", "7200"))
_HEALTH_POLL      = int(os.environ.get("PRECHECK_HEALTH_POLL", "15"))
_CHECK_TIMEOUT    = int(os.environ.get("PRECHECK_CHECK_TIMEOUT", "900"))
_BASE_DL_TIMEOUT  = os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200")


def container_name(competition_id: str) -> str:
    return f"tpn-precheck-{competition_id}"


def container_port(competition_id: str) -> int:
    digest = hashlib.sha256(competition_id.encode()).hexdigest()
    return _HOST_PORT_BASE + (int(digest, 16) % _HOST_PORT_RANGE)


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
        competition_id: Optional[str] = None,
        base_repo: Optional[str] = None,
        image: str = _IMAGE,
        health_timeout: int = _HEALTH_TIMEOUT,
        health_poll: int = _HEALTH_POLL,
        check_timeout: int = _CHECK_TIMEOUT,
    ):
        self._competition_id = competition_id
        self._base_repo = base_repo or ""
        self._image = image
        self._health_timeout = health_timeout
        self._health_poll = health_poll
        self._check_timeout = check_timeout
        self._name = container_name(competition_id) if competition_id else f"tpn-precheck-{uuid.uuid4().hex[:8]}"
        self._host_port = container_port(competition_id) if competition_id else _HOST_PORT_BASE
        self._container_name: Optional[str] = None
        self._base_url: str = f"http://localhost:{self._host_port}"

    def __enter__(self) -> "PrecheckContainer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def launch(self) -> None:
        if self._container_name:
            return
        if _docker_container_running(self._name):
            self._container_name = self._name
            return

        cmd = ["docker", "run", "-d", "--name", self._name, "-p", f"127.0.0.1:{self._host_port}:8080", "--rm",
               "-e", f"PRECHECK_LOG_LEVEL={LOG_LEVEL}"]
        if self._base_repo:
            cmd += ["-e", f"BASE_MODEL_REPO={self._base_repo}",
                    "-e", f"BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS={_BASE_DL_TIMEOUT}"]
        cmd.append(self._image)

        logger.info(f"Starting precheck container {self._name} (base_repo={self._base_repo or 'none'})")
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

        self._container_name = self._name
        logger.info(
            f"Container {self._name} started — "
            f"{'waiting for base model download' if self._base_repo else 'ready immediately'} "
            f"(up to {self._health_timeout}s)"
        )

    def start(self) -> None:
        self.launch()
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


def cleanup_stale_containers(keep_competition_ids: Optional[List[str]] = None) -> None:
    keep_names = {container_name(cid) for cid in (keep_competition_ids or [])}
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=^tpn-precheck-", "--format", "{{.ID}} {{.Names}}"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        logger.warning(f"Could not check for stale precheck containers: {e}")
        return

    stale_ids = []
    for line in result.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        container_id, name = parts
        if name not in keep_names:
            stale_ids.append(container_id)

    if not stale_ids:
        return

    logger.warning(f"Removing {len(stale_ids)} stale precheck container(s) from a prior run: {stale_ids}")
    try:
        subprocess.run(["docker", "rm", "-f", *stale_ids], capture_output=True, timeout=30)
    except Exception as e:
        logger.warning(f"Failed to remove stale precheck containers: {e}")


def _docker_container_running(name: str) -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", f"name=^{name}$"],
            capture_output=True, text=True, timeout=15,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False
    return bool(result.stdout.strip())


def is_container_up(competition_id: str) -> bool:
    name = container_name(competition_id)
    if not _docker_container_running(name):
        return False
    try:
        resp = requests.get(f"http://localhost:{container_port(competition_id)}/health", timeout=5)
        return resp.ok and bool(resp.json().get("ready"))
    except Exception:
        return False


def stop_container(competition_id: str) -> None:
    name = container_name(competition_id)
    logger.info(f"Stopping precheck container {name}")
    try:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
    except Exception as e:
        logger.warning(f"Failed to remove container {name}: {e}")


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
