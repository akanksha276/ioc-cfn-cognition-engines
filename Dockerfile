# ── Stage 1: install all dependencies into an isolated venv ─────────────────
FROM python:3.11.11-slim-bookworm AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
# Install Poetry (no venv needed for Poetry itself in the builder)
RUN pip install --no-cache-dir "poetry>=1.8.0" "poetry-plugin-export" && \
    poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project false && \
    poetry config virtualenvs.path /opt/venv-store

# Copy only the dependency manifest first (better layer caching)
COPY pyproject.toml ./
# poetry.lock is optional – copy if present
COPY poetry.loc[k] ./

# Resolve/generate the lock file, export to plain requirements, then install.
# fastembed (ONNX-based) replaces torch+sentence-transformers – no extra index needed.
RUN poetry lock --no-update 2>/dev/null || poetry lock
RUN poetry export \
    --without dev \
    --without-hashes \
    --format requirements.txt \
    -o /tmp/requirements-export.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements-export.txt

# Aggressive cleanup in a separate step so failures above are never masked
# Also strip packages that are transitive-only and not needed at inference time:
#   sympy     – onnxruntime shape-inference tooling, not used at runtime
#   hf_xet    – huggingface fast-download protocol, not needed post-install
#   pip       – not needed inside the runtime venv
#   setuptools – same
RUN find /opt/venv -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'tests'    -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'test'     -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -name '*.pyc' -delete 2>/dev/null || true \
    && find /opt/venv -name '*.pyo' -delete 2>/dev/null || true \
    && /opt/venv/bin/pip uninstall -y sympy hf_xet pip setuptools 2>/dev/null || true \
    && find /opt/venv -type d -name 'sympy'         -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name 'hf_xet'        -exec rm -rf {} + 2>/dev/null || true \
    && find /opt/venv -type d -name '*.dist-info' -name 'sympy*'   -exec rm -rf {} + 2>/dev/null || true

# ── Stage 2: download & quantize model (never reaches runtime) ───────────────
FROM python:3.11.11-slim-bookworm AS model-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir onnxruntime onnx sympy

# Download, quantize, and clean up — separate RUN steps are fine here since
# model-builder is intermediate; its layers never reach the final image
RUN mkdir -p /fastembed_cache/ibm-granite/granite-embedding-30m-english && \
    cd /fastembed_cache/ibm-granite/granite-embedding-30m-english && \
    curl --insecure --retry 5 --retry-delay 5 --retry-max-time 300 -fsSL -o config.json \
    "https://huggingface.co/ibm-granite/granite-embedding-30m-english/resolve/9b5b096411652ec1189c68fcfb90d0a82c5b45af/config.json" && \
    curl --insecure --retry 5 --retry-delay 5 --retry-max-time 300 -fsSL -o tokenizer.json \
    "https://huggingface.co/ibm-granite/granite-embedding-30m-english/resolve/9b5b096411652ec1189c68fcfb90d0a82c5b45af/tokenizer.json" && \
    curl --insecure --retry 5 --retry-delay 5 --retry-max-time 300 -fsSL -o tokenizer_config.json \
    "https://huggingface.co/ibm-granite/granite-embedding-30m-english/resolve/9b5b096411652ec1189c68fcfb90d0a82c5b45af/tokenizer_config.json" && \
    curl --insecure --retry 5 --retry-delay 5 --retry-max-time 300 -fsSL -o special_tokens_map.json \
    "https://huggingface.co/ibm-granite/granite-embedding-30m-english/resolve/9b5b096411652ec1189c68fcfb90d0a82c5b45af/special_tokens_map.json" && \
    curl --insecure --retry 5 --retry-delay 5 --retry-max-time 300 -fsSL -o model.onnx \
    "https://huggingface.co/ibm-granite/granite-embedding-30m-english/resolve/9b5b096411652ec1189c68fcfb90d0a82c5b45af/model.onnx"

RUN python3 -c "from onnxruntime.quantization import quantize_dynamic, QuantType; quantize_dynamic('/fastembed_cache/ibm-granite/granite-embedding-30m-english/model.onnx', '/fastembed_cache/ibm-granite/granite-embedding-30m-english/model_optimized.onnx', weight_type=QuantType.QUInt8)"

RUN rm /fastembed_cache/ibm-granite/granite-embedding-30m-english/model.onnx

# ── Stage 3: lean runtime image ──────────────────────────────────────────────
FROM python:3.11.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    EMBEDDING_MODEL_PATH=/app/granite-embedding-30m-english

# Install curl for Docker healthcheck (runtime only)
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv

# Copy quantized model from model-builder into fastembed cache
COPY --from=model-builder /fastembed_cache /tmp/fastembed_cache

# Bundle local model (avoids build-time download and SSL issues)
COPY --from=model-builder --chown=1000:1000 /fastembed_cache/ibm-granite/granite-embedding-30m-english/ /app/granite-embedding-30m-english/

# Copy each service into its own subdirectory (unified app runs from /app with PYTHONPATH=/app)
COPY --chown=1000:1000 common/                 /app/common/
COPY --chown=1000:1000 gateway/app/            /app/gateway/app/
COPY --chown=1000:1000 ingestion/app/          /app/ingestion/app/
COPY --chown=1000:1000 evidence/app/           /app/evidence/app/
COPY --chown=1000:1000 distill/app/            /app/distill/app/
COPY --chown=1000:1000 semantic_alignment/      /app/semantic_alignment/
COPY --chown=1000:1000 protocol/               /app/protocol/

RUN adduser --disabled-password --gecos "" --uid 1000 app 2>/dev/null || true

USER app

# Single process: unified app mounts ingestion, evidence, distill; one port (9004 per Confluence)
ENV PYTHONPATH=/app
EXPOSE 9004
WORKDIR /app
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9004/api/internal/diagnostics/health || exit 1
CMD ["uvicorn", "gateway.app.main:app", "--host", "0.0.0.0", "--port", "9004"]
