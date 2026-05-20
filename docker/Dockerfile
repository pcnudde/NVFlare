# syntax=docker/dockerfile:1.7

ARG PYTHON_BASE=python:3.14.5-slim-trixie@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856
ARG PIP_VERSION=26.1.1

FROM ${PYTHON_BASE} AS builder

ARG NVFL_EXPECT_VERSION=""
ARG NVFL_RELEASE=1
ARG PIP_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/NVFlare
COPY . .

RUN python -m pip install --upgrade "pip==${PIP_VERSION}" \
    && NVFL_RELEASE="${NVFL_RELEASE}" python -m pip wheel --wheel-dir /wheels ".[K8S]" \
    && python -c 'import glob, os, pathlib; expected = os.environ.get("NVFL_EXPECT_VERSION", ""); wheels = [pathlib.Path(p).name for p in glob.glob("/wheels/nvflare-*.whl")]; assert len(wheels) == 1, wheels; wheel = wheels[0]; print("Built NVFlare wheel:", wheel); assert not expected or f"-{expected}-" in wheel, f"expected {expected}, got {wheel}"'

FROM ${PYTHON_BASE}

ARG PIP_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends zip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/NVFlare

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels,readonly \
    python -m pip install --upgrade "pip==${PIP_VERSION}" \
    && python -m pip install --no-index --find-links=/wheels "nvflare[K8S]" \
    && python -m pip check
