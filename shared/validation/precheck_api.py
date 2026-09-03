#!/usr/bin/env python3
"""
Unified precheck API — runs inside a Docker container.

Startup:
  If BASE_MODEL_REPO env is set, downloads the base model to /base_model/ once.
  Sets ready=True once download completes (or immediately if no BASE_MODEL_REPO).
  /health returns {"ready": true} when ready.

Endpoints:
  GET  /health
    → {"ready": bool, "error": str|null}

  POST /check {"repository": "user/repo", "revision": "<sha>",
               "filename": "model.gguf", "context_length": 4096}
    Validator path. Downloads the GGUF from the Hub, runs all checks, deletes it.
    → PrecheckResponse

  POST /check-local {"path": "/data/model.gguf", "context_length": 4096}
    Miner self-check path. Reads from /data/ volume mount, skips provenance.
    → PrecheckResponse (provenance always null)

PrecheckResponse:
  {
    "provenance": {"is_derivative": bool, "cka": float, "notes": [...]} | null,
    "ram":        {"passed": bool, "ram_bytes": int, "weights_bytes": int, "kv_cache_bytes": int} | null,
    "sha256":     str | null,
    "error":      str | null
  }
"""
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as ApiModel
    import uvicorn
except ImportError:
    print("ERROR: pip install fastapi uvicorn", file=sys.stderr)
    sys.exit(2)

try:
    from huggingface_hub import hf_hub_download, snapshot_download
except ImportError:
    print("ERROR: pip install huggingface_hub", file=sys.stderr)
    sys.exit(2)

sys.path.insert(0, str(Path(__file__).parent))
from provenance_check import check_provenance

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_MODEL_REPO: str = os.environ.get("BASE_MODEL_REPO", "")
BASE_MODEL_DIR: Path = Path("/base_model")
MAX_MODEL_BYTES: int = int(os.environ.get("MAX_MODEL_BYTES", str(50 * 1024 ** 3)))  # 50 GiB
BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200"))
CKA_THRESHOLD: float = float(os.environ.get("CKA_THRESHOLD", "0.80"))
N_TOKENS: int = int(os.environ.get("N_TOKENS", "4096"))
RAM_CHECK_TIMEOUT_SECONDS: int = int(os.environ.get("RAM_CHECK_TIMEOUT_SECONDS", "600"))
LLAMA_THREADS: int = int(os.environ.get("LLAMA_THREADS", "8"))
PRECHECK_LOG_LEVEL: str = os.environ.get("PRECHECK_LOG_LEVEL", "INFO")
HF_TOKEN: str = os.environ.get("HF_TOKEN", "")


_STARTED = time.monotonic()


def _log(msg: str, level: str = "INFO") -> None:
    """Timestamped stderr line: wall-clock UTC + seconds since process start."""
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"{stamp} +{time.monotonic() - _STARTED:7.1f}s {level}: {msg}",
          file=sys.stderr, flush=True)


def _debug(msg: str) -> None:
    if PRECHECK_LOG_LEVEL == "DEBUG":
        _log(msg, "DEBUG")

# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

_ready = threading.Event()
_startup_error: str = ""


def _download_base_model() -> None:
    global _startup_error
    if not BASE_MODEL_REPO:
        _ready.set()
        return

    _log(f"Downloading base model: {BASE_MODEL_REPO} -> {BASE_MODEL_DIR}")
    started = time.monotonic()
    try:
        BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=BASE_MODEL_REPO,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.md", ".gitattributes", "*.txt"],
            token=HF_TOKEN or None,
        )
        elapsed = max(time.monotonic() - started, 1e-6)
        ggufs = list(BASE_MODEL_DIR.rglob("*.gguf"))
        if not ggufs:
            _startup_error = f"No .gguf files found in {BASE_MODEL_REPO}"
            _log(_startup_error, "ERROR")
            return
        total = sum(f.stat().st_size for f in ggufs)
        _log(f"Base model ready: {ggufs[0]} ({total:,} bytes in {elapsed:.1f}s, "
             f"{total / elapsed / 1e6:.1f} MB/s)")
        _ready.set()
    except Exception as e:
        _startup_error = f"Base model download failed after {time.monotonic() - started:.1f}s: {e}"
        _log(_startup_error, "ERROR")


@asynccontextmanager
async def lifespan(app):
    def _with_timeout():
        global _startup_error
        t = threading.Thread(target=_download_base_model, daemon=True)
        t.start()
        t.join(timeout=BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS)
        if t.is_alive() and not _ready.is_set():
            _startup_error = (
                f"base model download timed out after {BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS}s"
            )
            _log(_startup_error, "ERROR")
    threading.Thread(target=_with_timeout, daemon=True).start()
    yield


app = FastAPI(title="TPN Precheck API", docs_url=None, redoc_url=None, lifespan=lifespan)


class CheckRequest(ApiModel):
    repository: str          # bare HF repo id, "user/repo"
    revision: str            # immutable commit SHA
    filename: str            # file within the repo, e.g. "model.gguf"
    context_length: int = 4096


class CheckLocalRequest(ApiModel):
    path: str
    context_length: int = 4096


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"ready": _ready.is_set(), "error": _startup_error or None}


@app.post("/check")
def check(req: CheckRequest) -> JSONResponse:
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail=_startup_error or "not ready")

    tmp_dir = tempfile.mkdtemp(prefix="candidate-")
    try:
        _log(f"Downloading {req.repository}@{req.revision[:12]}/{req.filename}")
        started = time.monotonic()
        try:
            gguf_path = _download_hf(req.repository, req.revision, req.filename, tmp_dir)
        except _DownloadError as e:
            raise HTTPException(
                status_code=422,
                detail=f"download failed after {time.monotonic() - started:.1f}s: {e}",
            )

        elapsed = max(time.monotonic() - started, 1e-6)
        size = Path(gguf_path).stat().st_size
        _log(f"Downloaded {size:,} bytes in {elapsed:.1f}s ({size / elapsed / 1e6:.1f} MB/s)")

        if size > MAX_MODEL_BYTES:
            raise HTTPException(
                status_code=422,
                detail=f"file too large: {size:,} bytes > cap {MAX_MODEL_BYTES:,}",
            )

        return JSONResponse(_run_checks(gguf_path, req.context_length, run_provenance=True))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/check-local")
def check_local(req: CheckLocalRequest) -> JSONResponse:
    """Miner self-check: read from /data/ volume mount, skip provenance."""
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail=_startup_error or "not ready")

    # Constrain to /data/ to prevent path traversal
    resolved = Path(req.path).resolve()
    data_root = Path("/data").resolve()
    if not str(resolved).startswith(str(data_root)):
        raise HTTPException(status_code=400, detail="path must be under /data/")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"file not found: {resolved}")

    return JSONResponse(_run_checks(str(resolved), req.context_length, run_provenance=False))


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

def _run_checks(gguf_path: str, context_length: int, run_provenance: bool) -> dict:
    try:
        return _run_checks_inner(gguf_path, context_length, run_provenance)
    except Exception as e:
        _log(f"Checks failed unexpectedly for {gguf_path}: {e}", "ERROR")
        return {"provenance": None, "ram": None, "sha256": None, "error": f"precheck error: {e}"}


def _run_checks_inner(gguf_path: str, context_length: int, run_provenance: bool) -> dict:
    provenance_result = None
    ram_result = None
    error = None

    size_bytes = Path(gguf_path).stat().st_size
    checks_started = time.monotonic()
    _log(f"Running checks on {gguf_path} ({size_bytes:,} bytes, "
         f"context_length={context_length}, provenance={run_provenance})")

    # Provenance check
    if run_provenance and BASE_MODEL_REPO:
        base_ggufs = sorted(BASE_MODEL_DIR.rglob("*.gguf"))
        if not base_ggufs:
            error = "base model GGUF missing"
        else:
            _debug(f"Provenance check: base={base_ggufs[0]} candidate={gguf_path}")
            phase_started = time.monotonic()
            try:
                result = check_provenance(
                    path_a=str(base_ggufs[0]),
                    path_b=gguf_path,
                    threshold=CKA_THRESHOLD,
                    n_tokens=N_TOKENS,
                )
                notes: list = []
                if result.tokenizer:
                    notes.extend(result.tokenizer.notes)
                if result.embeddings:
                    notes.extend(result.embeddings.notes)
                provenance_result = {
                    "is_derivative": result.is_derivative,
                    "cka": result.embeddings.cka if result.embeddings else 0.0,
                    "notes": notes,
                }
                _log(f"Provenance done in {time.monotonic() - phase_started:.1f}s: "
                     f"is_derivative={result.is_derivative} cka={provenance_result['cka']:.3f}")
                _debug(f"Provenance notes={notes}")
            except Exception as e:
                error = f"provenance check error after {time.monotonic() - phase_started:.1f}s: {e}"

    # RAM check (always, even if provenance errored — independent gate)
    _debug(f"Starting RAM check (context_length={context_length})")
    phase_started = time.monotonic()
    try:
        ram_result, ram_reason = _run_llama_cli(gguf_path, context_length)
        _log(f"RAM check done in {time.monotonic() - phase_started:.1f}s: {ram_result}")
        if ram_reason and not error:
            error = f"ram check failed: {ram_reason}"
    except Exception as e:
        ram_result = _failed_ram()
        if not error:
            error = f"ram check error after {time.monotonic() - phase_started:.1f}s: {e}"

    # sha256 (always, independent of provenance/ram outcome)
    sha256_result = None
    phase_started = time.monotonic()
    try:
        _debug("Computing sha256")
        sha256_result = _sha256_file(gguf_path)
        _log(f"sha256 done in {time.monotonic() - phase_started:.1f}s: {sha256_result}")
    except Exception as e:
        if not error:
            error = f"sha256 check error: {e}"

    _log(f"Checks complete for {gguf_path} in {time.monotonic() - checks_started:.1f}s total: "
         f"error={error}")
    return {"provenance": provenance_result, "ram": ram_result, "sha256": sha256_result, "error": error}


# ---------------------------------------------------------------------------
# llama.cpp RAM measurement
# ---------------------------------------------------------------------------

_RE_WEIGHTS = re.compile(r"load_tensors:.*CPU model buffer size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)
_RE_KV      = re.compile(r"llama_kv_cache:.*CPU KV buffer size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)


def _failed_ram() -> dict:
    return {"passed": False, "ram_bytes": 0, "weights_bytes": 0, "kv_cache_bytes": 0}


def _run_llama_cli(gguf_path: str, context_length: int) -> tuple[dict, str | None]:
    """Returns (ram_result, reason). reason is None on success, else why it failed."""
    # --single-turn: exit after one response (b10020 stays as server otherwise)
    # --no-warmup: skip warmup pass to avoid SIGABRT on encoder-only models
    # findall[-1]: model loads twice internally; last match has real values (not 0.00 MiB)
    cmd = [
        "llama-cli",
        "-m", gguf_path,
        "-ngl", "0",
        "-c", str(context_length),
        "-np", "1",
        "-b", "2048",
        "-ub", "512",
        "-ctk", "f16",
        "-ctv", "f16",
        "-t", str(LLAMA_THREADS),
        "--no-mmap",
        "--verbose",
        "--single-turn",
        "--no-warmup",
        "-p", "hi",
    ]
    _debug(f"llama-cli command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=RAM_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        reason = f"llama-cli timed out after {RAM_CHECK_TIMEOUT_SECONDS}s (context_length={context_length})"
        _log(reason, "ERROR")
        return _failed_ram(), reason

    # llama-cli's generated output (--verbose logs the inference response) can
    # contain non-UTF-8 bytes from garbage token decodes; only the weight/KV
    # buffer-size log lines matter here, so replace rather than fail on those.
    combined = (result.stdout + result.stderr).decode("utf-8", errors="replace")

    if result.returncode != 0:
        # negative returncode == killed by signal (e.g. -9 SIGKILL from OOM killer)
        sig_note = f" (killed by signal {-result.returncode})" if result.returncode < 0 else ""
        reason = f"llama-cli exit {result.returncode}{sig_note}: {combined[-500:]}"
        _log(reason, "ERROR")
        return _failed_ram(), reason

    w_matches = _RE_WEIGHTS.findall(combined)
    kv_matches = _RE_KV.findall(combined)

    if not w_matches or not kv_matches:
        missing = "weights" if not w_matches else "kv-cache"
        reason = f"llama-cli log parse failed (missing {missing} buffer line)"
        _log(f"{reason}. tail:\n{combined[-1000:]}", "ERROR")
        return _failed_ram(), reason

    weights_bytes = int(float(w_matches[-1]) * 1024 * 1024)
    kv_bytes      = int(float(kv_matches[-1]) * 1024 * 1024)
    _debug(f"Parsed weights_bytes={weights_bytes:,} kv_cache_bytes={kv_bytes:,}")
    return {
        "passed": True,
        "ram_bytes": weights_bytes + kv_bytes,
        "weights_bytes": weights_bytes,
        "kv_cache_bytes": kv_bytes,
    }, None


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

class _DownloadError(Exception):
    pass


PROGRESS_INTERVAL_SECONDS = 15.0


def _log_progress(done: threading.Event, dest_dir: str) -> None:
    """
    Log bytes-on-disk every PROGRESS_INTERVAL_SECONDS until `done` is set.

    Watches the destination rather than hooking the hub's tqdm bars: the hub
    runs several bars (CAS fetch and file reconstruction) whose counters do not
    agree, and tqdm's carriage-return redraw is unreadable in `docker logs`
    anyway. Bytes landing on disk is the one number that means what it says.
    """
    started = time.monotonic()
    while not done.wait(PROGRESS_INTERVAL_SECONDS):
        try:
            size = sum(f.stat().st_size for f in Path(dest_dir).rglob("*") if f.is_file())
        except OSError:
            continue
        elapsed = max(time.monotonic() - started, 1e-6)
        _log(f"  downloading: {size:,} bytes ({size / elapsed / 1e6:.1f} MB/s avg)")


def _download_hf(repository: str, revision: str, filename: str, dest_dir: str) -> str:
    """
    Fetch one file from the Hub into dest_dir; returns its local path.

    Uses huggingface_hub rather than a hand-rolled stream: it downloads in
    parallel ranged chunks, retries on 429 using the RateLimit header, and
    resumes partial transfers. A single long-lived HTTP stream against the
    /resolve/ endpoint degrades badly on multi-GB files.
    """
    done = threading.Event()
    threading.Thread(target=_log_progress, args=(done, dest_dir), daemon=True).start()
    try:
        return hf_hub_download(
            repo_id=repository,
            filename=filename,
            revision=revision,
            local_dir=dest_dir,
            token=HF_TOKEN or None,
        )
    except Exception as e:
        raise _DownloadError(f"{type(e).__name__}: {e}")
    finally:
        done.set()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
