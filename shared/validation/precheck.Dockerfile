# Build for target platform:
#   amd64 (prod Linux):  docker build .
#   arm64 (Mac dev):     docker build --platform linux/arm64 .
FROM python:3.12-slim

WORKDIR /app

# llama.cpp prebuilt CPU binary — selects correct archive for build platform.
# Pin tag: b10020 — verify log line format against this build before changing.
ARG LLAMA_TAG=b10020
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        tar \
        libgomp1 \
    && if [ "$TARGETARCH" = "arm64" ]; then \
         LLAMA_TGZ="llama-${LLAMA_TAG}-bin-ubuntu-arm64.tar.gz"; \
       else \
         LLAMA_TGZ="llama-${LLAMA_TAG}-bin-ubuntu-x64.tar.gz"; \
       fi \
    && curl -fsSL \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_TAG}/${LLAMA_TGZ}" \
        -o /tmp/llama.tar.gz \
    && tar -xzf /tmp/llama.tar.gz -C /opt \
    && ln -s /opt/llama-${LLAMA_TAG}/llama-cli /usr/local/bin/llama-cli \
    && echo "/opt/llama-${LLAMA_TAG}" > /etc/ld.so.conf.d/llama.conf \
    && ldconfig \
    && rm -rf /tmp/llama.tar.gz \
    && rm -rf /var/lib/apt/lists/*

# Python deps
RUN pip install --no-cache-dir \
        fastapi==0.115.0 \
        uvicorn[standard]==0.32.0 \
        requests==2.32.3 \
        gguf>=0.19.0 \
        numpy>=2.2.0 \
        huggingface_hub>=1.2.0

COPY provenance_check.py precheck_api.py ./

EXPOSE 8080

# start-period covers optional base model download (up to 2h for large models).
# Reports unhealthy only on hard error, not mere not-ready, so orchestrators don't
# kill a container that is still downloading.
HEALTHCHECK --interval=30s --timeout=10s --start-period=7200s --retries=3 \
    CMD curl -sf http://localhost:8080/health | python3 -c \
        "import sys,json; d=json.load(sys.stdin); sys.exit(1 if d.get('error') else 0)"

ENV PORT=8080

ENTRYPOINT ["python", "precheck_api.py"]
