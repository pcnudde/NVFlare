FROM python:3.12-slim

WORKDIR /opt/nvflare
COPY . /opt/nvflare

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e .

ENV PYTHONUNBUFFERED=1
