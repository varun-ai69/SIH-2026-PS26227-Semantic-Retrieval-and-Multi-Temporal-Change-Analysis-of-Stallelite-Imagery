# 🚀 CANOPUS — Complete Teammate Onboarding & System Setup Guide

> **Project Name**: CANOPUS — Semantic Retrieval & Multi-Temporal Satellite Image Analysis  
> **Problem Statement**: SIH-26227  
> **Target Audience**: Team members, Evaluators & New Developers  

---

## 📋 Table of Contents
1. [System Prerequisites](#1-system-prerequisites)
2. [Hardware Recommendations](#2-hardware-recommendations)
3. [Step-by-Step Quick Setup (Docker - Recommended)](#3-step-by-step-quick-setup-docker---recommended)
4. [AI Model Checkpoints & Placement (Crucial)](#4-ai-model-checkpoints--placement-crucial)
5. [Environment Variables (`.env`)](#5-environment-variables-env)
6. [Starting & Stopping the Application](#6-starting--stopping-the-application)
7. [Service Access Endpoints (URLs)](#7-service-access-endpoints-urls)
8. [Verifying Your Setup (Smoke Tests)](#8-verifying-your-setup-smoke-tests)
9. [Troubleshooting & Common Issues](#9-troubleshooting--common-issues)

---

## 1. System Prerequisites

Before starting, ensure your system has the following software installed:

| Software | Version | Purpose | Download Link |
|---|---|---|---|
| **Git** | 2.30+ | Source code version control | [git-scm.com](https://git-scm.com/) |
| **Docker Desktop** | 4.25+ | Container virtualization runtime | [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/) |
| **Docker Compose** | v2.20+ | Multi-container orchestration (included in Docker Desktop) | — |

> ⚠️ **Windows Users**: Ensure **WSL 2 (Windows Subsystem for Linux)** is enabled in Docker Desktop under `Settings -> General -> Use the WSL 2 based engine`.

---

## 2. Hardware Recommendations

| Component | Minimum Specification | Recommended Specification |
|---|---|---|
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free disk space | 25 GB free SSD space |
| **CPU** | 4 Cores (Intel i5 / AMD Ryzen 5) | 8 Cores (Intel i7 / Ryzen 7 / Apple M-series) |
| **GPU (Optional)** | None (runs on CPU) | NVIDIA GPU (CUDA 12.1+) for faster embeddings |

---

## 3. Step-by-Step Quick Setup (Docker - Recommended)

Follow these simple steps to get the entire platform up and running in **under 5 minutes**:

### Step 3.1: Clone the Repository
Open PowerShell / Terminal and run:
```bash
git clone https://github.com/varun-ai69/SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-.git
cd SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-
```

### Step 3.2: Create Your Environment File
Copy the provided `.env.example` template:
```bash
# Windows (PowerShell):
Copy-Item .env.example .env

# Linux / macOS (Bash):
cp .env.example .env
```
*(All default values in `.env` are pre-configured to work out of the box with zero manual edits required!)*

---

## 4. AI Model Checkpoints & Placement (Crucial)

CANOPUS uses **two core foundation models** for high-dimensional semantic search and change detection. 

> 💡 **Auto-Download Notice**: If you do **not** download the weights manually, **the system will automatically fetch them from Hugging Face on the first search/detection run**.  
> However, for faster startup or slow internet connections, you can download them beforehand and place them in the specified directories.

### Model 1: RemoteCLIP ViT-B-32 (Semantic Search & Multimodal Retrieval)
* **File Name**: `RemoteCLIP-ViT-B-32.pt`
* **File Size**: ~605 MB
* **Exact Target Path in Project**:
  ```text
  models/retrieval/RemoteCLIP-ViT-B-32.pt
  ```
* **Download Direct Link**:  
  [https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt](https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt)
* **Terminal Download Command**:
  ```bash
  # Windows PowerShell:
  New-Item -ItemType Directory -Force -Path models/retrieval
  curl.exe -L -o models/retrieval/RemoteCLIP-ViT-B-32.pt "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt"

  # Linux / Mac:
  mkdir -p models/retrieval
  curl -L -o models/retrieval/RemoteCLIP-ViT-B-32.pt "https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt"
  ```

---

### Model 2: Prithvi-EO-2.0-300M (IBM-NASA Multi-Spectral Foundation Model)
* **File Name**: `Prithvi_EO_V2_300M.pt`
* **File Size**: ~1.32 GB
* **Exact Target Path in Project**:
  ```text
  models/retrieval/prithvi/Prithvi_EO_V2_300M.pt
  ```
* **Download Direct Link**:  
  [https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt)
* **Terminal Download Command**:
  ```bash
  # Windows PowerShell:
  New-Item -ItemType Directory -Force -Path models/retrieval/prithvi
  curl.exe -L -o models/retrieval/prithvi/Prithvi_EO_V2_300M.pt "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt"

  # Linux / Mac:
  mkdir -p models/retrieval/prithvi
  curl -L -o models/retrieval/prithvi/Prithvi_EO_V2_300M.pt "https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt"
  ```

---

### Verify Folder Structure
Your `models/` folder should look like this:
```text
models/
└── retrieval/
    ├── RemoteCLIP-ViT-B-32.pt       # 605 MB (Model 1)
    └── prithvi/
        ├── config.json              # Model configuration (Tracked in Git)
        ├── prithvi_mae.py           # Model architecture (Tracked in Git)
        └── Prithvi_EO_V2_300M.pt    # 1.32 GB (Model 2)
```

---

## 5. Environment Variables (`.env`)

The `.env` file contains settings for the database, vector store, and AI models. Key settings:

```dotenv
# Port configuration
PORT=8000
API_HOST=0.0.0.0

# Database Credentials
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=eo_archive
POSTGRES_USER=eo_admin
POSTGRES_PASSWORD=eo_password

# Vector DB
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# Device Mode (Set 'cpu' or 'cuda')
DEVICE=cpu

# Online/Offline Flags (0 = allow model auto-downloads, 1 = strict offline)
OFFLINE_MODE=false
HF_HUB_OFFLINE=0
TRANSFORMERS_OFFLINE=0
```

---

## 6. Starting & Stopping the Application

### Launch the Complete Stack
Run this single command from the project root:
```bash
docker compose up -d --build
```
> **What Happens Behind the Scenes Automatically**:
> 1. **PostgreSQL 16 + PostGIS 3.4** initializes schemas and tables (`scenes`, `tiles`, `change_runs`, `analyst_decisions`, `search_feedback`).
> 2. **Qdrant Vector DB** launches and ensures collection `tile_embeddings` is ready.
> 3. **MinIO Object Storage** starts on ports 9000 & 9001.
> 4. **FastAPI Backend** starts with hot-reloading enabled on port 8000.
> 5. **Frontend Web Dashboards** are automatically served.

### Check Container Status
```bash
docker compose ps
```
All 4 containers (`eo_backend`, `eo_postgres`, `eo_qdrant`, `eo_minio`) should show `Up` or `healthy`.

### View Live Logs
```bash
# View backend logs:
docker logs -f eo_backend

# View all services:
docker compose logs -f
```

### Stop the Stack
```bash
docker compose down
```
*(Data in PostgreSQL and Qdrant is persistently saved in Docker named volumes, so nothing is lost when you stop the containers).*

---

## 7. Service Access Endpoints (URLs)

Once started, open any of the following URLs in your web browser:

| Dashboard / Module | URL | Description |
|---|---|---|
| 🛰️ **Geospatial Map Workspace** | [http://localhost:8000](http://localhost:8000) | Satellite Map, AOI Polygon Drawing & Ingestion |
| 🔍 **Semantic Search & Chat** | [http://localhost:8000/retrieval.html](http://localhost:8000/retrieval.html) | Natural language text search & Image-to-Image similarity |
| 🔄 **Multi-Temporal Change Analysis** | [http://localhost:8000/change.html](http://localhost:8000/change.html) | Bi-temporal $T_1$ vs $T_2$ change detection & PDF dossier export |
| 🌌 **HDBSCAN Clustering Workbench** | [http://localhost:8000/clustering.html](http://localhost:8000/clustering.html) | Unsupervised terrain pattern discovery & medoid extraction |
| ⚖️ **Analyst Review & Audit Queue** | [http://localhost:8000/review.html](http://localhost:8000/review.html) | Human-in-the-loop candidate verification & audit logging |
| 📖 **FastAPI Swagger API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive REST API testing documentation |
| 🎯 **Qdrant Vector Console** | [http://localhost:6333/dashboard](http://localhost:6333/dashboard) | Vector embeddings visualization |
| 🪣 **MinIO S3 Web Console** | [http://localhost:9001](http://localhost:9001) | S3 storage (User: `eo_admin`, Password: `eo_password`) |

---

## 8. Verifying Your Setup (Smoke Tests)

### Test 1: Check Archive Statistics API
Open your browser or terminal and curl:
```bash
curl http://localhost:8000/api/v1/archive/stats
```
*Expected Output*: Returns JSON containing `total_tiles`, `total_scenes`, and ingested regions.

### Test 2: Run Automated Pytest Suite
Verify that all unit and integration tests pass inside the container:
```bash
docker exec -it eo_backend pytest tests/test_bugfixes.py tests/test_phase1_8_encoder_vectorstore.py -v
```

---

## 9. Troubleshooting & Common Issues

### Issue 1: Port Conflict (e.g. `port 8000 or 5434 is already in use`)
* **Cause**: Another service or previous container is using port 8000 or 5434.
* **Fix**: Open `.env` and change `PORT=8001` or `POSTGRES_PORT=5435`, then run:
  ```bash
  docker compose down
  docker compose up -d --build
  ```

### Issue 2: Docker Desktop RAM / Memory Limitation
* **Symptom**: Container gets killed with `Exit Code 137` (Out of Memory).
* **Fix**: Open Docker Desktop `Settings -> Resources -> Advanced` and set Memory to at least **6 GB** (preferably **8 GB**).

### Issue 3: Windows Line Endings (`\r\n` error on `entrypoint.sh`)
* **Symptom**: `standard_init_linux.go: exec user process caused: no such file or directory`
* **Fix**: The Dockerfile already contains `RUN sed -i 's/\r$//' /app/infra/docker/entrypoint.sh`, but if you edit shell scripts on Windows, run:
  ```bash
  git config core.autocrlf input
  ```

---

*Team CANOPUS — Ready for Evaluation & Deployment.* 🎯
