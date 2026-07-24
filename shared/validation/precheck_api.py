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

  POST /check {"url": "https://...", "context_length": 4096}
    Validator path. Downloads submitted GGUF, runs all checks, deletes it.
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
import subprocess
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel as ApiModel
    import uvicorn
    import requests as http_requests
except ImportError:
    print("ERROR: pip install fastapi uvicorn requests", file=sys.stderr)
    sys.exit(2)

try:
    from huggingface_hub import snapshot_download
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
DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "600"))
BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200"))
CKA_THRESHOLD: float = float(os.environ.get("CKA_THRESHOLD", "0.80"))
N_TOKENS: int = int(os.environ.get("N_TOKENS", "4096"))
RAM_CHECK_TIMEOUT_SECONDS: int = int(os.environ.get("RAM_CHECK_TIMEOUT_SECONDS", "600"))
LLAMA_THREADS: int = int(os.environ.get("LLAMA_THREADS", "8"))

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

    print(f"Downloading base model: {BASE_MODEL_REPO} -> {BASE_MODEL_DIR}", flush=True)
    try:
        BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=BASE_MODEL_REPO,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.md", ".gitattributes", "*.txt"],
        )
        ggufs = list(BASE_MODEL_DIR.rglob("*.gguf"))
        if not ggufs:
            _startup_error = f"No .gguf files found in {BASE_MODEL_REPO}"
            print(f"ERROR: {_startup_error}", file=sys.stderr)
            return
        print(f"Base model ready: {ggufs[0]}", flush=True)
        _ready.set()
    except Exception as e:
        _startup_error = f"Base model download failed: {e}"
        print(f"ERROR: {_startup_error}", file=sys.stderr)


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
            print(f"ERROR: {_startup_error}", file=sys.stderr)
    threading.Thread(target=_with_timeout, daemon=True).start()
    yield


app = FastAPI(title="TPN Precheck API", docs_url=None, redoc_url=None, lifespan=lifespan)


class CheckRequest(ApiModel):
    url: str
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

    url = req.url.strip()
    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="url must start with https://")

    tmp_path: str = ""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gguf")
        os.close(tmp_fd)
        try:
            _download_url(url, tmp_path)
        except _DownloadError as e:
            raise HTTPException(status_code=422, detail=f"download failed: {e}")

        return JSONResponse(_run_checks(tmp_path, req.context_length, run_provenance=True))
    finally:
        _delete(tmp_path)


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
    provenance_result = None
    ram_result = None
    error = None

    # Provenance check
    if run_provenance and BASE_MODEL_REPO:
        base_ggufs = sorted(BASE_MODEL_DIR.rglob("*.gguf"))
        if not base_ggufs:
            error = "base model GGUF missing"
        else:
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
            except Exception as e:
                error = f"provenance check error: {e}"

    # RAM check (always, even if provenance errored — independent gate)
    try:
        ram_result = _run_llama_cli(gguf_path, context_length)
    except Exception as e:
        ram_result = {"passed": False, "ram_bytes": 0, "weights_bytes": 0, "kv_cache_bytes": 0}
        if not error:
            error = f"ram check error: {e}"

    # sha256 (always, independent of provenance/ram outcome)
    sha256_result = None
    try:
        sha256_result = _sha256_file(gguf_path)
    except Exception as e:
        if not error:
            error = f"sha256 check error: {e}"

    return {"provenance": provenance_result, "ram": ram_result, "sha256": sha256_result, "error": error}


# ---------------------------------------------------------------------------
# llama.cpp RAM measurement
# ---------------------------------------------------------------------------

_RE_WEIGHTS = re.compile(r"load_tensors:.*CPU model buffer size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)
_RE_KV      = re.compile(r"llama_kv_cache:.*CPU KV buffer size\s*=\s*([\d.]+)\s*MiB", re.IGNORECASE)


def _run_llama_cli(gguf_path: str, context_length: int) -> dict:
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

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=RAM_CHECK_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return {"passed": False, "ram_bytes": 0, "weights_bytes": 0, "kv_cache_bytes": 0}

    combined = result.stdout + result.stderr

    if result.returncode != 0:
        print(f"llama-cli exit {result.returncode}: {combined[-500:]}", file=sys.stderr)
        return {"passed": False, "ram_bytes": 0, "weights_bytes": 0, "kv_cache_bytes": 0}

    w_matches = _RE_WEIGHTS.findall(combined)
    kv_matches = _RE_KV.findall(combined)

    if not w_matches or not kv_matches:
        print(f"llama-cli log parse failed. tail:\n{combined[-1000:]}", file=sys.stderr)
        return {"passed": False, "ram_bytes": 0, "weights_bytes": 0, "kv_cache_bytes": 0}

    weights_bytes = int(float(w_matches[-1]) * 1024 * 1024)
    kv_bytes      = int(float(kv_matches[-1]) * 1024 * 1024)
    return {
        "passed": True,
        "ram_bytes": weights_bytes + kv_bytes,
        "weights_bytes": weights_bytes,
        "kv_cache_bytes": kv_bytes,
    }


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

class _DownloadError(Exception):
    pass


def _download_url(url: str, dest: str) -> None:
    try:
        resp = http_requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, allow_redirects=True)
    except Exception as e:
        raise _DownloadError(f"request failed: {e}")

    if resp.status_code == 404:
        raise _DownloadError("file not found (404)")
    if resp.status_code in (401, 403):
        raise _DownloadError(f"access denied ({resp.status_code}) — repo must be public")
    if not resp.ok:
        raise _DownloadError(f"HTTP {resp.status_code}")

    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_MODEL_BYTES:
        raise _DownloadError(f"file too large: {int(content_length):,} bytes > cap {MAX_MODEL_BYTES:,}")

    written = 0
    try:
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                written += len(chunk)
                if written > MAX_MODEL_BYTES:
                    raise _DownloadError(f"download exceeded cap {MAX_MODEL_BYTES:,} bytes mid-stream")
                f.write(chunk)
    except _DownloadError:
        raise
    except Exception as e:
        raise _DownloadError(f"write error: {e}")


def _delete(path: str) -> None:
    if path and Path(path).exists():
        try:
            Path(path).unlink()
        except Exception:
            pass


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
