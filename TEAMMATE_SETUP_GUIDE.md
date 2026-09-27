# 🛰️ CANOPUS — Quick Setup Guide

Follow these simple steps to run the entire system locally:

---

### Step 1: Clone the Repository
```bash
git clone https://github.com/varun-ai69/SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-.git
cd SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-
```

---

### Step 2: Create the Environment File
Copy `.env.example` to `.env`:

**Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**Linux / macOS:**
```bash
cp .env.example .env
```
*(No edits required — all defaults work out of the box).*

---

### Step 3: Download & Place Model Weights (2 Files)

Download both files and place them in the following paths:

#### 1. RemoteCLIP ViT-B-32 (~605 MB)
* **Download**: [RemoteCLIP-ViT-B-32.pt](https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt)
* **Save to**: `models/retrieval/RemoteCLIP-ViT-B-32.pt`

#### 2. Prithvi-EO-2.0 (~1.32 GB)
* **Download**: [Prithvi_EO_V2_300M.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt)
* **Save to**: `models/retrieval/prithvi/Prithvi_EO_V2_300M.pt`

**Expected `models/` directory structure:**
```text
models/
└── retrieval/
    ├── RemoteCLIP-ViT-B-32.pt
    └── prithvi/
        ├── config.json
        ├── prithvi_mae.py
        └── Prithvi_EO_V2_300M.pt
```

---

### Step 4: Start the System with Docker
Make sure Docker Desktop is running, then execute:
```bash
docker compose up -d --build
```

---

### 🚀 Access the Web Applications
Once started, open these in your browser:
* **Map & Ingestion**: [http://localhost:8000](http://localhost:8000)
* **Semantic Search & Chat**: [http://localhost:8000/retrieval.html](http://localhost:8000/retrieval.html)
* **Change Detection & PDF Export**: [http://localhost:8000/change.html](http://localhost:8000/change.html)
* **Clustering & Discovery**: [http://localhost:8000/clustering.html](http://localhost:8000/clustering.html)
* **Analyst Review Queue**: [http://localhost:8000/review.html](http://localhost:8000/review.html)
* **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### 🛑 Stop the System
```bash
docker compose down
```
