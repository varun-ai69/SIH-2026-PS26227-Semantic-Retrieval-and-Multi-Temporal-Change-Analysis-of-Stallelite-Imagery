# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for GDAL, Geos, PostGIS, Proj
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgdal-dev \
    gdal-bin \
    libgeos-dev \
    libproj-dev \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 && \
    pip install -r requirements.txt

# Copy source repository
COPY . /app/
RUN sed -i 's/\r$//' /app/infra/docker/entrypoint.sh && chmod +x /app/infra/docker/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/bin/bash", "/app/infra/docker/entrypoint.sh"]
