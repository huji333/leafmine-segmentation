FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv (manages Python itself)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock README.md .python-version ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install Python 3.13 + dependencies via uv
RUN uv sync --frozen

# Default command
CMD ["uv", "run", "python", "-c", "print('leafmine-segmentation ready')"]
