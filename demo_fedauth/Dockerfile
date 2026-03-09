FROM python:3.12-slim

WORKDIR /opt/nvflare
COPY pyproject.toml setup.py setup.cfg MANIFEST.in versioneer.py README.md /opt/nvflare/
COPY nvflare /opt/nvflare/nvflare
COPY nvflight /opt/nvflare/nvflight
COPY job_templates /opt/nvflare/job_templates

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
