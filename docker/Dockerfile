# Changed to -runtime base image (lighter weight)
FROM runpod/pytorch:0.7.2-dev-cu1241-torch260-ubuntu2204 AS base

ARG HUGGING_FACE_HUB_TOKEN

# Environment variables
ENV HF_HOME=/workspace/cache/huggingface \
    PIP_CACHE_DIR=/workspace/cache/pip \
    DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1 \
    WORKDIR=/app
    TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0 10.0 12.0"

# Combined system setup
RUN apt update && apt install -y --no-install-recommends \
    git wget screen python3.11-venv libvips unzip nano htop nvtop \
    && apt clean \
    && rm -rf /var/lib/apt/lists/* \
    && echo "if [ -f /workspace/.bashrc ]; then source /workspace/.bashrc; fi" >> /root/.bashrc \
    && pip install --upgrade pip

# Node.js 23.x installation (as requested)
RUN curl -fsSL https://deb.nodesource.com/setup_23.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ai-toolkit
WORKDIR /app
RUN git clone https://github.com/ostris/ai-toolkit.git \
    && cd ai-toolkit \
    && git checkout main

WORKDIR /app/ai-toolkit

# build tools & requirements.txt
RUN pip install --no-cache-dir \
    poetry-core==1.9.0 \
    setuptools==69.5.1 \
    --pre torch torchvision --index-url https://download.pytorch.org/whl/cu124 --force

RUN pip install --no-cache-dir \
    -r requirements.txt

# build UI components
RUN cd ui && \
    npm install && \
    npm run build && \
    npm run update_db

# Workspace setup
RUN mkdir -p /app/workspace && chmod -R 755 /app/workspace

# Final setup
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 8675 8888
WORKDIR /
CMD ["/start.sh"]
