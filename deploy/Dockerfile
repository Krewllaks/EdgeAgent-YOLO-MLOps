# EdgeAgent — NVIDIA GPU Container
# Build:  docker build -f deploy/Dockerfile -t edgeagent .
# Run:    docker compose -f deploy/docker-compose.yml up -d

FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# System deps
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1 \
    curl && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY deploy/requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Copy application
COPY main.py .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create data directories
RUN mkdir -p data/audit data/feedback data/shifts logs models/registry

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

EXPOSE 8080 4840

ENTRYPOINT ["python", "main.py"]
CMD ["--config", "configs/production_config.yaml"]
