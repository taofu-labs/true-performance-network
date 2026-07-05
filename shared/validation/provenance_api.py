#!/usr/bin/env python3
"""
Provenance check HTTP API — runs inside a Docker container.

Startup:
  Downloads the base model (BASE_MODEL_REPO env) once to /base_model/.
  Sets ready=True once download completes; /health returns {"ready": true}.

Endpoints:
  GET  /health          -> {"ready": bool}
  POST /check           -> {"url": "https://hf.co/..."}
                       -> {"is_derivative": bool, "cka": float, "notes": [...]}
                       -> {"error": "..."} on failure

The submitted GGUF is downloaded to a temp file, checked against base,
then deleted. Download size is capped at MAX_MODEL_BYTES (default 50 GiB).
"""
import os
import sys
import tempfile
import threading
from pathlib import Path

# --- deps: fastapi + uvicorn installed by Dockerfile, not in pyproject ---
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
# Timeout for each submitted-GGUF download (connect + read per chunk, not total transfer).
# A slow server can stall for up to this long before the request errors.
DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("DOWNLOAD_TIMEOUT_SECONDS", "600"))  # 10 min
# Total wall-clock budget for the base model snapshot_download on startup.
# Large quantised models (e.g. 70B Q4 ~40 GB) can take 30-60 min on a slow link.
BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS: int = int(os.environ.get("BASE_MODEL_DOWNLOAD_TIMEOUT_SECONDS", "7200"))  # 2h
CKA_THRESHOLD: float = float(os.environ.get("CKA_THRESHOLD", "0.80"))
N_TOKENS: int = int(os.environ.get("N_TOKENS", "4096"))

# ---------------------------------------------------------------------------
# App + state
# ---------------------------------------------------------------------------

app = FastAPI(title="TPN Provenance API", docs_url=None, redoc_url=None)

_ready = threading.Event()
_startup_error: str = ""


class CheckRequest(ApiModel):
    url: str


# ---------------------------------------------------------------------------
# Startup: download base model
# ---------------------------------------------------------------------------

def _download_base_model() -> None:
    global _startup_error
    if not BASE_MODEL_REPO:
        _startup_error = "BASE_MODEL_REPO env not set"
        print(f"ERROR: {_startup_error}", file=sys.stderr)
        return

    print(f"Downloading base model: {BASE_MODEL_REPO} -> {BASE_MODEL_DIR}", flush=True)
    try:
        BASE_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=BASE_MODEL_REPO,
            local_dir=str(BASE_MODEL_DIR),
            ignore_patterns=["*.md", ".gitattributes", "*.txt"],
        )
        # Verify at least one GGUF landed
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


@app.on_event("startup")
def startup_event() -> None:
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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict:
    return {"ready": _ready.is_set(), "error": _startup_error or None}


@app.post("/check")
def check(req: CheckRequest) -> JSONResponse:
    if not _ready.is_set():
        raise HTTPException(status_code=503, detail=_startup_error or "base model not ready")

    url = req.url.strip()
    if not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="url must start with https://")

    # Find base GGUF
    base_ggufs = sorted(BASE_MODEL_DIR.rglob("*.gguf"))
    if not base_ggufs:
        raise HTTPException(status_code=500, detail="base model GGUF missing")
    base_path = base_ggufs[0]

    # Download submitted GGUF to a temp file
    tmp_path: str = ""
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".gguf")
        os.close(tmp_fd)

        try:
            _download_url(url, tmp_path)
        except _DownloadError as e:
            raise HTTPException(status_code=422, detail=f"download failed: {e}")

        # Run provenance check
        try:
            result = check_provenance(
                path_a=str(base_path),
                path_b=tmp_path,
                threshold=CKA_THRESHOLD,
                n_tokens=N_TOKENS,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"provenance check error: {e}")

        notes: list = []
        if result.tokenizer:
            notes.extend(result.tokenizer.notes)
        if result.embeddings:
            notes.extend(result.embeddings.notes)

        return JSONResponse({
            "is_derivative": result.is_derivative,
            "cka": result.embeddings.cka if result.embeddings else 0.0,
            "notes": notes,
        })

    finally:
        # Always delete submitted file
        if tmp_path and Path(tmp_path).exists():
            try:
                Path(tmp_path).unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

class _DownloadError(Exception):
    pass


def _download_url(url: str, dest: str) -> None:
    """Stream download url to dest. Enforces MAX_MODEL_BYTES cap."""
    try:
        resp = http_requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, allow_redirects=True)
    except Exception as e:
        raise _DownloadError(f"request failed: {e}")

    if resp.status_code == 404:
        raise _DownloadError("file not found (404)")
    if resp.status_code == 401 or resp.status_code == 403:
        raise _DownloadError(f"access denied ({resp.status_code}) — repo must be public")
    if not resp.ok:
        raise _DownloadError(f"HTTP {resp.status_code}")

    content_length = resp.headers.get("Content-Length")
    if content_length and int(content_length) > MAX_MODEL_BYTES:
        raise _DownloadError(
            f"file too large: {int(content_length):,} bytes > cap {MAX_MODEL_BYTES:,}"
        )

    written = 0
    try:
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                written += len(chunk)
                if written > MAX_MODEL_BYTES:
                    raise _DownloadError(
                        f"download exceeded cap {MAX_MODEL_BYTES:,} bytes mid-stream"
                    )
                f.write(chunk)
    except _DownloadError:
        raise
    except Exception as e:
        raise _DownloadError(f"write error: {e}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
