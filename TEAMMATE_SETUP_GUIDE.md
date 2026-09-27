# 🛰️ CANOPUS — Quick Setup Guide for Teammates

Sirf yeh **4 steps** follow karne hain:

---

### Step 1: Clone Repo
```bash
git clone https://github.com/varun-ai69/SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-.git
cd SIH-2026-PS26227-Semantic-Retrieval-and-Multi-Temporal-Analysis-
```

---

### Step 2: Environment File Banao
```bash
# Windows (PowerShell):
Copy-Item .env.example .env

# Linux / Mac:
cp .env.example .env
```
*(Kuch bhi edit nahi karna, defaults direct chalenge)*

---

### Step 3: Models Download & Place Karo (2 Files)

Dono files download karke exact in folders mein daal do:

#### 1. RemoteCLIP (~605 MB)
* **Download Link**: [RemoteCLIP-ViT-B-32.pt](https://huggingface.co/chendelong/RemoteCLIP/resolve/main/RemoteCLIP-ViT-B-32.pt)
* **Kha Rakhna Hai**: `models/retrieval/RemoteCLIP-ViT-B-32.pt`

#### 2. Prithvi-EO-2.0 (~1.32 GB)
* **Download Link**: [Prithvi_EO_V2_300M.pt](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M/resolve/main/Prithvi_EO_V2_300M.pt)
* **Kha Rakhna Hai**: `models/retrieval/prithvi/Prithvi_EO_V2_300M.pt`

> **Final `models/` folder aisa dikhna chahiye:**
> ```text
> models/
> └── retrieval/
>     ├── RemoteCLIP-ViT-B-32.pt
>     └── prithvi/
>         ├── config.json
>         ├── prithvi_mae.py
>         └── Prithvi_EO_V2_300M.pt
> ```

---

### Step 4: Docker Run Karo
```bash
docker compose up -d --build
```

---

### 🚀 Done! Browser Mein Open Karo:
* **Map & Ingestion**: [http://localhost:8000](http://localhost:8000)
* **Semantic Search & Chat**: [http://localhost:8000/retrieval.html](http://localhost:8000/retrieval.html)
* **Change Detection & PDF Export**: [http://localhost:8000/change.html](http://localhost:8000/change.html)
* **Clustering**: [http://localhost:8000/clustering.html](http://localhost:8000/clustering.html)
* **Analyst Review Queue**: [http://localhost:8000/review.html](http://localhost:8000/review.html)

---

### 🛑 Stop Karna Ho Toh:
```bash
docker compose down
```
