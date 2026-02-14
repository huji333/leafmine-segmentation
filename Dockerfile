FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv (manages Python itself)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies only (skip building the local project)
COPY pyproject.toml uv.lock README.md .python-version ./
RUN uv sync --frozen --no-install-project

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Make leafmine_seg importable without package install
ENV PYTHONPATH=/app/src

# Default command
CMD ["uv", "run", "python", "-c", "print('leafmine-segmentation ready')"]
