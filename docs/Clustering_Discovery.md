# 🌐 Canopus Unsupervised Clustering & Cross-Site Discovery Specification

This document provides the complete technical specification for the **Unsupervised Clustering and Cross-Site Discovery Subsystem** (`backend/jobs/run_clustering.py`, `backend/services/discovery.py`, `backend/api/routers/discovery.py`).

It details how Canopus periodically organizes large-scale satellite archives into semantic clusters using **HDBSCAN** and empowers defense intelligence analysts to branch out from a single tile of interest to discover visual and contextual "twins" across the entire archive with zero manual query engineering.

---

## 1. Problem Statement Requirements Addressed (PS 2.2.4)

The Discovery and Clustering subsystem directly fulfills all mandates outlined in Ministry of Defence (MoD) / Indian Army Problem Statement **SIH-26227 Section 2.2.4**:

| PS 2.2.4 Mandate | Operational Capability | Pipeline Implementation |
|---|---|---|
| **Unsupervised Site Clustering** | Unsupervised, embedding-based clustering of similar geographic, terrain, and infrastructure sites across the entire archive. | Offline/periodic background job (`backend/jobs/run_clustering.py`) scrolls all 512-D RemoteCLIP vectors from Qdrant, normalizes to unit length, and executes **HDBSCAN** density-based clustering with medoid selection. |
| **Cross-Site Discovery ("Find Similar")** | Allow an analyst who identifies a single location of interest to branch out and discover other locations with comparable visual or semantic characteristics without constructing manual queries for each site. | Seed-based discovery service (`GET /api/v1/discover/{tile_id}`) pulls the seed tile's precomputed 512-D embedding directly from Qdrant, executing high-speed $k$-NN nearest-neighbor search across all regions with **zero prompt engineering and zero re-encoding**. |
| **Instant Cluster Member Exploration** | Inspect and browse all imagery tiles belonging to a common semantic partition group without query execution. | Cluster retrieval service (`GET /api/v1/clusters/{cluster_id}/tiles`) uses indexed PostgreSQL queries (`SELECT tile_id FROM tiles WHERE cluster_id = %s`) to instantly return member tiles. |
| **Tactical Terrain Explainability** | Human-interpretable classification of cluster themes. | Automated cluster labeling derives dominant terrain types (e.g., *"Dense Woodland & Forest Canopy"*, *"Industrial / Tarmac Logistics"*) from the average multi-spectral indices (average NDVI, NDWI, NDBI) of cluster members. |

---

## 2. System Architecture & Dual Operational Modes

The Discovery and Clustering subsystem operates in two complementary phases:
1. **Asynchronous Background Batch Job:** Periodic archive-wide partition recomputation.
2. **Synchronous Analyst Discovery API:** Real-time, single-click discovery from any candidate tile.

```
═════════════════════════════════════════════════════════════════════════════════════
PHASE 1: BACKGROUND CLUSTERING JOB (backend/jobs/run_clustering.py)
═════════════════════════════════════════════════════════════════════════════════════

   ┌────────────────────────────────────────────────────────┐
   │ Qdrant Vector Archive ('tile_embeddings', 'maxar_...') │
   └──────────────────────────┬─────────────────────────────┘
                              │ Paginated Scroll (200 pts/batch)
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ 512-D Embedding Extraction & Unit L2 Normalization     │
   └──────────────────────────┬─────────────────────────────┘
                              │
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ HDBSCAN Density-Based Clustering                       │
   │ (Adaptive KMeans Fallback if Archive < 4 Samples)      │
   └──────────────────────────┬─────────────────────────────┘
                              │ Cluster Labels (Outliers = -1)
                              ▼
   ┌────────────────────────────────────────────────────────┐
   │ Medoid Selection (Centroid-Closest Vector)             │
   │ & Spectral Heuristic Labeling via PostgreSQL           │
   └──────────────────────────┬─────────────────────────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
   ┌───────────────────────┐     ┌────────────────────────┐
   │ PostgreSQL Sync:      │     │ Qdrant Payload Sync:   │
   │ • UPDATE tiles        │     │ • set_payload(         │
   │   SET cluster_id=...  │     │     cluster_id=...     │
   │ • REFRESH clusters    │     │   )                    │
   └───────────────────────┘     └────────────────────────┘


═════════════════════════════════════════════════════════════════════════════════════
PHASE 2: ANALYST DISCOVERY & "FIND SIMILAR" (backend/services/discovery.py)
═════════════════════════════════════════════════════════════════════════════════════

   [Analyst Identifies Tile of Interest] ──► Clicks "Find Similar"
   (e.g., uncatalogued airstrip or depot)         │
                                                  ▼
                                      GET /api/v1/discover/{tile_id}
                                                  │
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ Direct Qdrant Vector Retrieve │
                                  │ (Zero Prompt / Zero Encoding) │
                                  └───────────────┬───────────────┘
                                                  │ 512-D Seed Vector
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ Global k-NN Vector Search     │
                                  │ (Cosine Distance Matching)    │
                                  └───────────────┬───────────────┘
                                                  │ Ranked Candidates
                                                  ▼
                                  ┌───────────────────────────────┐
                                  │ PostgreSQL Metadata Hydration │
                                  │ + Spatial Deduplication       │
                                  └───────────────┬───────────────┘
                                                  │
                                                  ▼
                                  [Ranked Semantic Twin Tiles with]
                                  [Spectral Indices & Footprints  ]
```

---

## 3. Asynchronous Clustering Pipeline Mechanics

The background clustering job (`backend/jobs/run_clustering.py`) is decoupled from real-time query handling and can be triggered via command line (`python -m backend.jobs.run_clustering`) or via REST API (`POST /api/v1/clusters/recompute`).

### 3.1 Embedding Harvesting & Normalization
1. **Multi-Collection Scroll:** Traverses both Sentinel-2 (`tile_embeddings`) and Maxar (`maxar_tile_embeddings`) Qdrant collections using paginated scrolling (`limit=200`) to retrieve all stored vector embeddings and corresponding `tile_id`s.
2. **$L_2$ Unit Normalization:** Normalizes all embedding vectors $X$ to unit length:
   $$\hat{X}_i = \frac{X_i}{\|X_i\|_2}$$
   In unit-normalized Euclidean space, Euclidean distance is monotonically related to Cosine distance:
   $$d_{\text{euclidean}}(\hat{u}, \hat{v})^2 = 2 - 2 \cos(\hat{u}, \hat{v})$$
   This guarantees that Euclidean-based density clustering directly clusters vectors by their visual/semantic cosine similarity.

### 3.2 HDBSCAN Density-Based Clustering
The pipeline applies **HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)**:
* **Dynamic Minimum Cluster Size:**
  ```python
  min_cluster_size = min(5, max(2, n_samples // 3))
  ```
* **Metric:** Euclidean distance over unit-normalized embeddings.
* **Noise / Outlier Handling:** Outlier tiles that do not conform to any cohesive cluster are assigned label `-1`. These points have their PostgreSQL and Qdrant `cluster_id` set to `NULL`, preventing false semantic groupings.
* **Adaptive KMeans Fallback:** If sample sizes are small ($n \ge 4$) and HDBSCAN groups all samples into noise or fewer than 2 clusters, the system automatically falls back to adaptive KMeans clustering with `k = min(5, max(2, n // 2))` to ensure clear partitioning.

### 3.3 Medoid Selection & Spectral Label Synthesis
For each coherent cluster $C_k$ containing member vectors:
1. **Centroid Computation:** Calculates the geometric mean vector of all member embeddings.
2. **Medoid (Representative Tile) Selection:** The actual tile whose vector is closest to the mathematical centroid is elected as the cluster's representative medoid. The medoid's thumbnail is displayed as the visual cover card for the cluster in the analyst UI.
3. **Spectral Signature Labeling:**
   The job queries PostgreSQL for the average spectral indices of all member tiles:
   ```python
   avg_ndvi = mean([tile.mean_ndvi for tile in cluster_members])
   avg_ndwi = mean([tile.mean_ndwi for tile in cluster_members])
   avg_ndbi = mean([tile.mean_ndbi for tile in cluster_members])
   ```
   Automated heuristics classify the cluster into military and tactical landscape categories:
   - If `avg_ndwi >= 0.15` $\implies$ **Aquatic & Coastal Water**
   - Else if `avg_ndvi >= 0.35` $\implies$ **Dense Woodland & Forest Canopy**
   - Else if `avg_ndvi >= 0.20` $\implies$ **Vegetative & Agricultural Fields**
   - Else if `avg_ndbi >= 0.03` $\implies$ **Urban Core & Built-up Infrastructure**
   - Else if `avg_ndbi >= 0.00` $\implies$ **Industrial / Tarmac Logistics**
   - Else $\implies$ **Open Arid & Transition Terrain**

### 3.4 Dual-Store Persistence & Atomic Synchronization
Once clusters are synthesized:
1. **PostgreSQL Batch Update:** Executes `psycopg2.extras.execute_batch` updating `tiles.cluster_id` in chunks of 500 records.
2. **Clusters Table Refresh:** Clears and populates the `clusters` table with `cluster_id`, `label`, `representative_tile_id`, `tile_count`, `computed_at`, and `model_version`.
3. **Qdrant Payload Synchronization:** Calls `client.set_payload(collection_name, payload={"cluster_id": c_id}, points=[point_id])` ensuring high-dimensional vector search can filter by cluster natively.

---

## 4. Analyst Discovery Workflow ("Find Similar")

When an intelligence analyst identifies a target location of interest (e.g., from a map view, a search result, or an alert), they do not need to formulate descriptive text queries or find reference imagery.

### 4.1 Zero-Prompt / Zero-Encoding Seed Discovery
1. **Existing Tile Identification:** The seed tile is already ingested and indexed in the archive with a unique `tile_id`.
2. **Direct Vector Lookup:** `DiscoveryService.get_tile_vector(tile_id)` queries Qdrant directly using the tile's deterministic UUIDv5:
   ```python
   point_uuid = tile_id_to_uuid(tile_id)
   pts = qdrant.retrieve(collection_name=coll, ids=[point_uuid], with_vectors=True)
   seed_vector = pts[0].vector
   ```
   **No neural network forward pass or GPU inference is required.** The 512-D vector is instantly fetched from RAM.
3. **Global Archive $k$-NN Traversal:** Qdrant performs nearest-neighbor search across all collections using Cosine distance:
   $$\text{Cosine Similarity}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
   The search automatically excludes the seed `tile_id` itself and enforces a minimum similarity cutoff (`min_similarity >= 0.40`).
4. **PostgreSQL Hydration & Spatial Deduplication:** Candidates are hydrated from PostgreSQL with coordinates, acquisition dates, cloud percentages, and spectral means, filtering out redundant overlapping patches within ~200m.

### 4.2 Instant Cluster Member Retrieval
If an analyst wants to inspect all tiles grouped under a cluster:
* The system executes a single index-accelerated SQL query:
  ```sql
  SELECT tile_id FROM tiles WHERE cluster_id = 'cluster_20260914023000_1' LIMIT 60;
  ```
* All member tiles across diverse dates and AOIs are returned immediately, allowing analysts to monitor all instances of a specific terrain or installation category across the nation.

---

## 5. REST API Contract & Schemas

### 5.1 Endpoints Summary

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/discover/{tile_id}` | Discovers visual & semantic twins across the archive for a given tile. |
| `GET` | `/api/v1/clusters` | Lists all discovered terrain & land-use clusters with medoids and statistics. |
| `GET` | `/api/v1/clusters/{cluster_id}/tiles` | Returns all member tiles belonging to a specific cluster. |
| `POST` | `/api/v1/clusters/recompute` | Triggers the background HDBSCAN clustering job. |

### 5.2 Endpoint Specifications

#### 1. Single-Tile Discovery (`GET /api/v1/discover/{tile_id}`)
* **Query Parameters:**
  - `top_k` (integer, default `10`, range `1..50`): Number of similar tiles to return.
  - `min_similarity` (float, default `0.40`, range `0.0..1.0`): Cosine similarity threshold.
* **Response:** Array of `SearchResultItem` objects:
  ```json
  [
    {
      "tile_id": "tile_s2_43RVM_20240315_0042",
      "similarity_score": 0.892,
      "thumbnail_url": "/data/tiles/tile_s2_43RVM_20240315_0042_thumb.jpg",
      "centroid_lat": 31.4285,
      "centroid_lon": 74.3120,
      "acquisition_date": "2024-03-15T05:30:00Z",
      "sensor": "Sentinel-2",
      "cloud_pct": 0.012,
      "quality_confidence": 0.985,
      "mean_ndvi": 0.12,
      "mean_ndwi": -0.05,
      "mean_ndbi": 0.28,
      "spot_description": "Structural/Urban Site (NDBI: 0.28, NDVI: 0.12)"
    }
  ]
  ```

#### 2. Discovered Clusters Summary (`GET /api/v1/clusters`)
* **Response:**
  ```json
  [
    {
      "cluster_id": "cluster_20260914023000_1",
      "label": "Cluster 1 — Industrial / Tarmac Logistics (NDVI 0.14, NDBI 0.08)",
      "representative_tile_id": "tile_s2_43RVM_20240315_0012",
      "tile_count": 48,
      "computed_at": "2026-09-14T02:30:00Z",
      "model_version": "RemoteCLIP-ViT-B-32",
      "thumbnail_url": "/data/tiles/tile_s2_43RVM_20240315_0012_thumb.jpg",
      "centroid_lat": 31.5492,
      "centroid_lon": 74.3436,
      "mean_ndvi": 0.14,
      "mean_ndwi": -0.08,
      "mean_ndbi": 0.08,
      "sensor": "Sentinel-2"
    }
  ]
  ```

#### 3. Recompute Clusters Job (`POST /api/v1/clusters/recompute`)
* **Response:**
  ```json
  {
    "status": "success",
    "total_points": 512,
    "clusters_count": 8,
    "outliers_count": 14,
    "elapsed_seconds": 3.42,
    "computed_at": "2026-09-14T02:30:00Z"
  }
  ```

---

## 6. Verification & Problem Statement Closure

This implementation provides complete verification and closure for **PS Section 2.2.4**:
- ✅ **Unsupervised Site Clustering:** Automatic grouping of imagery patches across all ingested regions using HDBSCAN density clustering without requiring manual training labels.
- ✅ **No Manual Query Overhead:** Analysts can instantly find identical tactical installations (e.g. forward air bases, ammunition depots, river crossings) across different geographical sectors by clicking "Find Similar" on any verified patch.
- ✅ **Zero Computation Latency:** Uses stored 512-D vectors in Qdrant; discovery requires zero neural network inference.
- ✅ **Auditable Groupings:** Every cluster features an elected medoid tile and an interpretable multi-spectral label grounded in physical reflectance indices ($\text{NDVI}, \text{NDWI}, \text{NDBI}$).
