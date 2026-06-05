FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        git \
        nodejs \
        npm \
        ripgrep \
    && rm -rf /var/lib/apt/lists/*

RUN npm install -g @openai/codex @anthropic-ai/claude-code

RUN python -m pip install --upgrade pip \
    && python -m pip install torch

RUN python -m pip install hermes-agent==0.15.1

RUN printf '%s\n' \
        '#!/bin/sh' \
        'export NVIDIA_BASE_URL="${NVIDIA_BASE_URL:-https://inference-api.nvidia.com/v1}"' \
        'exec hermes "$@"' \
        > /usr/local/bin/hermes-nvidia \
    && chmod +x /usr/local/bin/hermes-nvidia

RUN useradd -m dev

USER dev
ENV HOME=/home/dev
ENV PATH=/home/dev/.local/bin:$PATH
ENV PIP_USER=1

WORKDIR /workspace
