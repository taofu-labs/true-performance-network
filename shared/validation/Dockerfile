FROM python:3.12-slim

WORKDIR /app

# System deps for gguf / numpy
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# API + provenance deps
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.32.0 \
        requests==2.32.3 \
        gguf>=0.19.0 \
        numpy>=2.2.0 \
        scipy>=1.15.0 \
        huggingface_hub>=0.23.0

COPY provenance_check.py provenance_api.py ./

EXPOSE 8080

# start-period covers base model download; large models can take 30-60+ min.
# The validator polls /health itself — this HEALTHCHECK is for orchestrator visibility only,
# so we report unhealthy only on hard failure (error set), not mere not-ready.
HEALTHCHECK --interval=30s --timeout=10s --start-period=7200s --retries=3 \
    CMD curl -sf http://localhost:8080/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(1 if d.get('error') else 0)"

ENV PORT=8080

ENTRYPOINT ["python", "provenance_api.py"]
