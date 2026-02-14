FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.9 via deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Pin uv to use Python 3.9
ENV UV_PYTHON=python3.9
COPY .python-version /root/.python-version

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install dependencies
RUN uv sync --frozen

# Default command
CMD ["uv", "run", "python", "-c", "print('leafmine-segmentation ready')"]
