# PaperGuard -- Streamlit app + local deberta-v3-large AI detector, containerized
# for Alibaba Cloud (Function Compute 3.0 / PAI-EAS / plain ECS+Docker).
#
# Deliberate choices:
#   * python:3.11-slim base -- small, matches the project's tested runtime.
#   * torch is installed EXPLICITLY from the CPU-only wheel index. Plain `pip
#     install torch` from requirements.txt would otherwise pull the CUDA build
#     (multiple GB) even though the detector agent runs fine on CPU and no GPU
#     is assumed for the serving container -- this keeps the image well under
#     Alibaba FC 3.0's 30 GB container limit and avoids downloading a CUDA
#     toolkit nobody will use for a single-request Streamlit inference path.
#     If you DO deploy on a GPU instance (e.g. PAI-EAS with a GPU spec), build
#     with `--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121`
#     to get the matching CUDA wheels instead.
#   * requirements-train.txt is intentionally NOT installed (training-only,
#     per its own header comment) -- keeps the serving image lean.
#   * Listens on $PORT (default 8000) with server.address=0.0.0.0, matching
#     the serverless-container convention Alibaba FC / most PaaS expect.
#   * Runs as a non-root user.

FROM python:3.11-slim

ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

WORKDIR /app

# System deps: PyMuPDF/pymupdf4llm need minimal build tooling on some
# platforms; kept small and removed in the same layer.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install torch (CPU wheels by default) as its own layer so it caches
# independently of the rest of requirements.txt.
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX_URL} torch

COPY requirements.txt .
# torch is already installed above; avoid re-resolving/overwriting it here.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Deployment images don't need training artifacts, caches, or dev/test files
# (also excluded via .dockerignore, but a second guard here doesn't hurt if
# someone builds from a dirty context).
RUN rm -rf training hf_cache .cache tests *.log

RUN useradd --create-home --uid 1000 paperguard \
    && chown -R paperguard:paperguard /app
USER paperguard

ENV PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf_cache \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8000

# Basic liveness check: Streamlit exposes /_stcore/health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import os,urllib.request; urllib.request.urlopen(f'http://127.0.0.1:{os.environ[\"PORT\"]}/_stcore/health').read()" || exit 1

# Shell form so $PORT expands; server.address=0.0.0.0 is required for the
# container's exposed port to actually be reachable from outside.
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0 --server.headless=true
