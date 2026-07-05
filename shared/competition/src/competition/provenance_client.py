"""
Validator-side manager for the dockerised provenance-check API.

Lifecycle (per competition scoring run):
    with ProvenanceContainer(base_repo, image) as ctr:
        verdict = ctr.check(gguf_download_url)

Or manually:
    ctr = ProvenanceContainer(base_repo)
    ctr.start()          # docker run -d, waits until /health ready
    verdict = ctr.check(url)
    ctr.stop()           # docker rm -f

Config (env):
    PROVENANCE_IMAGE          Docker image tag  (default: tpn-provenance)
    PROVENANCE_HOST_PORT      Host port to bind  (default: 8081)
    PROVENANCE_HEALTH_TIMEOUT Wall-clock seconds to wait for base model ready (default: 7200)
    PROVENANCE_HEALTH_POLL    Seconds between /health polls  (default: 15)
    PROVENANCE_CHECK_TIMEOUT  HTTP timeout for /check calls (default: 300 — CKA can be slow)
    BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS  Forwarded to container env (default: 7200)
"""
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import List, Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_IMAGE         = os.environ.get("PROVENANCE_IMAGE", "tpn-provenance")
_HOST_PORT     = int(os.environ.get("PROVENANCE_HOST_PORT", "8081"))
_HEALTH_TIMEOUT = int(os.environ.get("PROVENANCE_HEALTH_TIMEOUT", "7200"))   # 2 h
_HEALTH_POLL   = int(os.environ.get("PROVENANCE_HEALTH_POLL", "15"))
_CHECK_TIMEOUT = int(os.environ.get("PROVENANCE_CHECK_TIMEOUT", "300"))
_BASE_DL_TIMEOUT = os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceVerdict:
    is_derivative: bool
    cka: float = 0.0
    notes: List[str] = field(default_factory=list)
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Container manager
# ---------------------------------------------------------------------------

class ProvenanceContainer:
    """
    Manages one Docker container running provenance_api.py.

    Thread-safe for sequential use within a single scoring run.
    Not safe for concurrent calls to check() — one check at a time.
    """

    def __init__(
        self,
        base_repo: str,
        image: str = _IMAGE,
        host_port: int = _HOST_PORT,
        health_timeout: int = _HEALTH_TIMEOUT,
        health_poll: int = _HEALTH_POLL,
        check_timeout: int = _CHECK_TIMEOUT,
    ):
        self._base_repo = base_repo
        self._image = image
        self._host_port = host_port
        self._health_timeout = health_timeout
        self._health_poll = health_poll
        self._check_timeout = check_timeout
        self._container_name: Optional[str] = None
        self._base_url: str = f"http://localhost:{host_port}"

    # ── Context manager ────────────────────────────────────────────────

    def __enter__(self) -> "ProvenanceContainer":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> None:
        """Start container and block until /health reports ready or timeout."""
        if self._container_name:
            raise RuntimeError("ProvenanceContainer already started")

        name = f"tpn-provenance-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-p", f"{self._host_port}:8080",
            "-e", f"BASE_MODEL_REPO={self._base_repo}",
            "-e", f"BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS={_BASE_DL_TIMEOUT}",
            "--rm",
            self._image,
        ]

        logger.info(f"Starting provenance container {name} (base={self._base_repo})")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except subprocess.TimeoutExpired:
            raise RuntimeError("docker run timed out after 30s")
        except FileNotFoundError:
            raise RuntimeError("docker not found — install Docker or set PROVENANCE_IMAGE")

        if result.returncode != 0:
            raise RuntimeError(
                f"docker run failed (exit {result.returncode}): {result.stderr.strip()}"
            )

        self._container_name = name
        logger.info(f"Container {name} started — waiting for base model download (up to {self._health_timeout}s)")
        self._wait_until_ready()

    def stop(self) -> None:
        """Remove the container. Safe to call even if start() failed."""
        if not self._container_name:
            return
        name = self._container_name
        self._container_name = None
        logger.info(f"Stopping provenance container {name}")
        try:
            subprocess.run(
                ["docker", "rm", "-f", name],
                capture_output=True, timeout=15,
            )
        except Exception as e:
            logger.warning(f"Failed to remove container {name}: {e}")

    # ── Check ──────────────────────────────────────────────────────────

    def check(self, gguf_download_url: str) -> ProvenanceVerdict:
        """
        POST the submitted GGUF download URL to the container.
        Container downloads the file, runs provenance check, deletes it, returns verdict.
        Returns ProvenanceVerdict with error set if anything goes wrong.
        """
        if not self._container_name:
            return ProvenanceVerdict(is_derivative=False, error="container not running")

        try:
            resp = requests.post(
                f"{self._base_url}/check",
                json={"url": gguf_download_url},
                timeout=self._check_timeout,
            )
        except requests.Timeout:
            return ProvenanceVerdict(
                is_derivative=False,
                error=f"provenance /check timed out after {self._check_timeout}s",
            )
        except requests.ConnectionError as e:
            return ProvenanceVerdict(is_derivative=False, error=f"connection error: {e}")

        if resp.status_code == 503:
            return ProvenanceVerdict(is_derivative=False, error="container not ready (503)")

        try:
            data = resp.json()
        except Exception:
            return ProvenanceVerdict(
                is_derivative=False,
                error=f"non-JSON response {resp.status_code}: {resp.text[:200]}",
            )

        if not resp.ok:
            detail = data.get("detail") or data.get("error") or resp.text[:200]
            return ProvenanceVerdict(is_derivative=False, error=f"API error {resp.status_code}: {detail}")

        return ProvenanceVerdict(
            is_derivative=bool(data.get("is_derivative", False)),
            cka=float(data.get("cka", 0.0)),
            notes=list(data.get("notes", [])),
        )

    # ── Internal ───────────────────────────────────────────────────────

    def _wait_until_ready(self) -> None:
        """Poll /health until ready=true, error set, or timeout."""
        deadline = time.monotonic() + self._health_timeout
        last_log = 0.0

        while time.monotonic() < deadline:
            try:
                resp = requests.get(f"{self._base_url}/health", timeout=5)
                if resp.ok:
                    data = resp.json()
                    if data.get("ready"):
                        logger.info(f"Provenance container ready (base model loaded)")
                        return
                    if data.get("error"):
                        raise RuntimeError(f"Provenance container startup failed: {data['error']}")
            except requests.ConnectionError:
                pass  # container still booting
            except RuntimeError:
                raise
            except Exception as e:
                logger.debug(f"Health poll error (will retry): {e}")

            now = time.monotonic()
            if now - last_log >= 60:
                elapsed = int(now - (deadline - self._health_timeout))
                logger.info(f"Waiting for provenance container... {elapsed}s elapsed")
                last_log = now

            time.sleep(self._health_poll)

        raise RuntimeError(
            f"Provenance container not ready after {self._health_timeout}s "
            f"(base model download may still be running)"
        )
