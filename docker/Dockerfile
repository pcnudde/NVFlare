# syntax=docker/dockerfile:1.7

ARG PYTHON_BUILD_BASE=python:3.14.5-slim-trixie@sha256:7a500125bc50693f2214e842a621440a1b1b9cbb2188f74ab045d29ed2ea5856
ARG PYTHON_RUNTIME_BASE=nvcr.io/nvidia/distroless/python:3.14-v4.0.7@sha256:49751a52c1b4f59e0b68d6caf6728f305afc9e47c507008f8a9e8e1253929676
ARG PYTHON_VERSION=3.14
ARG PIP_VERSION=26.1.1
ARG KUBERNETES_VERSION=35.0.0
ARG NVFL_PACKAGE_NAME=nvflare

FROM ${PYTHON_BUILD_BASE} AS builder

ARG NVFL_EXPECT_VERSION=""
ARG KUBERNETES_VERSION
ARG NVFL_PACKAGE_NAME
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

RUN printf "kubernetes==%s\n" "${KUBERNETES_VERSION}" > /tmp/container-constraints.txt \
    && python -m pip install --upgrade "pip==${PIP_VERSION}" \
    && NVFL_RELEASE="${NVFL_RELEASE}" python -m pip wheel --wheel-dir /wheels -c /tmp/container-constraints.txt ".[K8S]" \
    && python -c 'import glob, os, pathlib; expected = os.environ.get("NVFL_EXPECT_VERSION", ""); package = os.environ.get("NVFL_PACKAGE_NAME", "nvflare").replace("-", "_"); wheels = [pathlib.Path(p).name for p in glob.glob(f"/wheels/{package}-*.whl")]; assert len(wheels) == 1, wheels; wheel = wheels[0]; print("Built NVFlare wheel:", wheel); assert not expected or f"-{expected}-" in wheel, f"expected {expected}, got {wheel}"'

FROM ${PYTHON_BUILD_BASE} AS runtime-tools

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends zip \
    && rm -rf /var/lib/apt/lists/*

FROM ${PYTHON_BUILD_BASE} AS installer

ARG NVFL_PACKAGE_NAME
ARG KUBERNETES_VERSION
ARG PIP_VERSION
ARG PYTHON_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/opt/nvflare/lib/python${PYTHON_VERSION}/site-packages \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN --mount=type=bind,from=builder,source=/wheels,target=/wheels,readonly \
    printf "kubernetes==%s\n" "${KUBERNETES_VERSION}" > /tmp/container-constraints.txt \
    && python -m pip install --upgrade "pip==${PIP_VERSION}" \
    && python -m pip install --prefix=/opt/nvflare --no-index --find-links=/wheels -c /tmp/container-constraints.txt "${NVFL_PACKAGE_NAME}[K8S]" \
    && python -c 'import importlib.metadata as m, os; expected = os.environ.get("KUBERNETES_VERSION"); actual = m.version("kubernetes"); print("Installed kubernetes:", actual); assert actual == expected, f"expected {expected}, got {actual}"' \
    && python -m pip check \
    && python -c 'import os, pathlib, sysconfig; site = pathlib.Path(sysconfig.get_paths()["purelib"]); site.mkdir(parents=True, exist_ok=True); (site / "nvflare-prefix.pth").write_text("/opt/nvflare/lib/python%s/site-packages\n" % os.environ["PYTHON_VERSION"])'

FROM ${PYTHON_RUNTIME_BASE}

ARG PYTHON_VERSION

ENV PATH=/opt/nvflare/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    PYTHONPATH=/opt/nvflare/lib/python${PYTHON_VERSION}/site-packages:/usr/local/lib/python${PYTHON_VERSION}/site-packages:/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY --from=installer /opt/nvflare /opt/nvflare
COPY --from=installer /usr/local/lib/python${PYTHON_VERSION}/site-packages/nvflare-prefix.pth /usr/local/lib/python${PYTHON_VERSION}/site-packages/nvflare-prefix.pth
COPY --from=runtime-tools /usr/bin/zip /usr/bin/zip

CMD ["python"]
