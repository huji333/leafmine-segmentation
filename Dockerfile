FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install uv (manages Python itself)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install Python
COPY .python-version ./
RUN uv python install

# Install dependencies via pip (avoid uv sync hang)
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project -o requirements.txt && \
    uv pip install --system -r requirements.txt

# Copy source
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Make leafmine_seg importable without package install
ENV PYTHONPATH=/app/src

# Default command
CMD ["uv", "run", "python", "-c", "print('leafmine-segmentation ready')"]
