# GhostLM training environment (issue #17)
#
# Reproducible Python + PyTorch environment for running GhostLM training,
# eval, and the supporting scripts. Targets CPU + CUDA. M4 / MPS users
# should run natively rather than via Docker (Apple Silicon GPU is not
# exposed inside Linux containers).
#
# Build:
#   docker build -t ghostlm:v0.6.0 .
#
# Run training (CPU):
#   docker run --rm -v $(pwd):/workspace ghostlm:v0.6.0 \
#     make train-tiny
#
# Run with CUDA:
#   docker run --rm --gpus all -v $(pwd):/workspace ghostlm:v0.6.0 \
#     python scripts/train.py --preset ghost-small --device cuda

FROM python:3.11-slim

# System deps. git is needed by some HF tooling; build-essential is for
# wheels that don't ship pre-compiled (datasets, sentencepiece occasionally).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Pinned dependencies first so the layer cache hits even when source changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Source last, so iterating on code does not invalidate the dependency layer.
COPY . .

# Default to a quick health check rather than starting training, so an
# accidental `docker run` does not eat resources.
CMD ["python", "-c", "import torch; from ghostlm.config import GhostLMConfig; print('GhostLM container ready'); print('torch:', torch.__version__, 'cuda:', torch.cuda.is_available())"]
