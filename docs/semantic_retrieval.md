# 🔍 Canopus Multimodal Semantic Retrieval Pipeline Specification

This document provides the complete technical specification for the **Canopus Multimodal Semantic Retrieval Pipeline** (`backend/services/vector_search.py`, `backend/services/encoder.py`, `backend/api/routers/search.py`).

It details how natural language text prompts and exemplar satellite image patches are encoded into a shared vision-language latent space, matched against high-dimensional vector collections using $k$-NN search, filtered by spatial and temporal parameters, and enriched with PostgreSQL spatial metadata.

---

## 1. Problem Statement Requirements Addressed (PS 2.2.1)

The Semantic Retrieval Pipeline directly fulfills all functional requirements specified in Ministry of Defence (MoD) / Indian Army Problem Statement **SIH-26227 Section 2.2.1**:

| PS 2.2.1 Mandate | Operational Capability | Pipeline Implementation |
|---|---|---|
| **Free-Text Search** | Search satellite imagery archives using natural language queries without prior coordinates or timestamps. | Fine-tuned `RemoteCLIP ViT-B-32` text encoder maps prompts (e.g. *"airfield runways with hangars near water"*) into 512-dimensional unit vectors. |
| **Multimodal Image-to-Image Search** | Upload a reference satellite image patch to find visually and contextually similar ground locations. | RemoteCLIP vision encoder processes uploaded image bytes (JPEG, PNG, GeoTIFF) into the identical 512-D vector space. |
| **Rank-Ordered Results** | Rank candidate ground locations by mathematical similarity score. | Cosine vector similarity matching (score in [0.0, 1.0]) in Qdrant; results returned in descending relevance order. |
| **Compound Filtering** | Refine semantic queries by geographic area, time window, sensor, and quality parameters. | Hybrid query engine combines PostGIS spatial intersection (`ST_Intersects`) with Qdrant payload filters (sensor, acquisition date, cloud %, quality score). |
| **Analyst Inspection & Explainability** | Provide verifiable evidence, indices, and contextual explanations for retrieved candidates. | Hydrates tiles from PostgreSQL with true footprints, spectral index distributions (NDVI, NDWI, NDBI), and automated tactical spot descriptions. |

---

## 2. End-to-End Retrieval Architecture

The pipeline bridges unstructured human language and high-resolution Earth Observation rasters through a shared 512-dimensional vector space:

```
                                  QUERY PHASE
                                  ───────────
   [Analyst Text Query]                     [Exemplar Image Upload]
("runways near river")                     (Reference .jpg / .tif)
           │                                          │
           ▼                                          ▼
 [RemoteCLIP Text Encoder]                 [RemoteCLIP Vision Encoder]
 (open_clip ViT-B-32)                      (open_clip ViT-B-32)
           │                                          │
           └───────────────────┬──────────────────────┘
                               │
                               ▼
                   [512-D L2-Normalized Vector]
                               │
                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │         Qdrant Vector Database (k-NN / HNSW Search)      │
   │  Collections: 'tile_embeddings' & 'maxar_tile_embeddings'│
   ├──────────────────────────────────────────────────────────┤
   │  Compound Pre-Filters Applied:                           │
   │  • Spatial AOI: Candidate tile_ids from PostGIS          │
   │  • Sensor: Sentinel-2 / Maxar WorldView                  │
   │  • Temporal: Date range [start_date, end_date]           │
   │  • Quality Gate: quality_confidence >= min_quality       │
   │  • Cloud Cover: cloud_pct <= max_cloud_pct               │
   └───────────────────────────┬──────────────────────────────┘
                               │ Top-K Candidates (tile_id, score)
                               ▼
   ┌──────────────────────────────────────────────────────────┐
   │        PostgreSQL + PostGIS Single-Query Hydration       │
   ├──────────────────────────────────────────────────────────┤
   │  • Preserves exact Qdrant similarity rank order          │
   │  • Enforces spatial deduplication (~200m / site_key)     │
   │  • Fetches geometries, spectral indices, and file paths  │
   │  • Synthesizes multi-index rule-based spot descriptions  │
   └───────────────────────────┬──────────────────────────────┘
                               │
                               ▼
             [Rank-Ordered Analyst Result Cards & Map HUD]
```

---

## 3. Dual Query Modalities & Feature Encoding (`backend/services/encoder.py`)

Canopus utilizes **RemoteCLIP** (a foundation model fine-tuned specifically on remote sensing image-text pairs) to align visual features with tactical military vocabulary.

### 3.1 Singleton Encoder Service (`RemoteCLIPEncoder`)
- **Model Architecture:** `ViT-B-32` (Vision Transformer backbone with 32×32 patch size).
- **Weight Checkpoint:** Local air-gapped checkpoint `models/retrieval/RemoteCLIP-ViT-B-32.pt` (~605 MB).
- **Embedding Dimensionality:** $D = 512$.
- **Normalization:** All vectors are strictly $L_2$-normalized:
  $$\mathbf{v}_{\text{norm}} = \frac{\mathbf{v}}{\|\mathbf{v}\|_2} \quad \implies \quad \|\mathbf{v}_{\text{norm}}\|_2 = 1.0$$

### 3.2 Modality A: Natural Language Text Search
1. When an analyst enters a search query (e.g. *"coastal naval base with docked ships"* or *"ammunition storage bunkers on arid terrain"*), the text is tokenized with `open_clip.get_tokenizer("ViT-B-32")`.
2. The transformer text encoder processes tokens and extracts 512-dimensional semantic embeddings via `encode_text()`.
3. The resulting vector resides in the exact same metric space as the satellite image tiles.

### 3.3 Modality B: Image-to-Image Visual Similarity Search
1. When an analyst uploads an exemplar satellite patch (JPEG, PNG, or GeoTIFF):
   - `_to_pil_image()` validates and converts the input stream into a 3-channel RGB image.
   - `preprocess(img)` applies standard bicubic resampling, center cropping to $224 \times 224$, and channel normalization.
2. The Vision Transformer processes the image tensor through multi-head self-attention layers via `encode_image()`.
3. The extracted 512-D vector captures structural geometry, texture, and spectral appearance, enabling visual "twin" discovery.

---

## 4. Vector Database Matching & Compound Filtering (`backend/services/vector_search.py`)

Vector retrieval is executed through `VectorSearchService.search_vectors()`, orchestrating high-speed Approximate Nearest Neighbor ($k$-NN) matching in Qdrant.

### 4.1 Similarity Metric
Because vectors are $L_2$-normalized, the **Cosine Similarity** equals the inner dot product:
$$\text{Similarity}(\mathbf{q}, \mathbf{t}) = \frac{\mathbf{q} \cdot \mathbf{t}}{\|\mathbf{q}\|_2 \|\mathbf{t}\|_2} = \mathbf{q} \cdot \mathbf{t} = \sum_{i=1}^{512} q_i \cdot t_i$$
Matches are scored between $0.0$ (dissimilar) and $1.0$ (identical).

### 4.2 Compound Filtering Mechanics (`build_qdrant_filter`)
Unlike basic vector databases that can only filter after finding neighbors, Qdrant applies **payload pre-filtering** during graph traversal:

1. **Spatial AOI Pre-Filtering (`resolve_aoi_to_tile_ids`):**
   - If the analyst specifies a spatial boundary (GeoJSON polygon or bounding box `[min_lon, min_lat, max_lon, max_lat]`), the service queries PostGIS first:
     ```sql
     SELECT tile_id FROM tiles
     WHERE ST_Intersects(geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326));
     ```
   - The intersecting candidate `tile_id`s are injected directly into Qdrant's filter conditions (`MatchAny(any=candidate_tile_ids)`). Only vectors from that exact geographic sector are traversed.
2. **Sensor Filtering:** Supports filtering by `"Sentinel-2"`, `"Maxar WorldView"`, or `"All"`. The query automatically searches the corresponding Qdrant collections (`tile_embeddings` and/or `maxar_tile_embeddings`).
3. **Temporal Date-Range Filtering:** Converts ISO timestamps into UNIX seconds, filtering `acquisition_date` via range condition (`gte` / `lte`).
4. **Quality Gate Filter (`min_quality`):** Enforces a minimum `quality_confidence` threshold (e.g. $\ge 0.80$), screening out compromised observations.
5. **Cloud Cover Filter (`max_cloud_pct`):** Ensures returned tiles have cloud contamination below the analyst's limit (e.g. $\le 15.0\%$).

---

## 5. PostgreSQL Hydration, Spatial Deduplication & Tactical Spot Descriptions

Once Qdrant returns candidate `(tile_id, score)` pairs, `hydrate_from_postgres()` enriches each candidate with operational metadata in a single database roundtrip.

### 5.1 Single-Query Batch Hydration
```sql
SELECT
    t.tile_id, t.scene_id, t.site_key, t.acquisition_date, t.sensor,
    t.cloud_pct, t.quality_confidence, t.mean_ndvi, t.mean_ndwi, t.mean_ndbi,
    t.centroid_lat, t.centroid_lon, t.file_path, t.thumbnail_path,
    ST_AsGeoJSON(t.geometry)::json AS geometry_geojson
FROM tiles t
WHERE t.tile_id IN %s;
```
The result rows are mapped back to Qdrant's exact score ranking order.

### 5.2 Spatial Deduplication
In dense archives, a single physical airfield or port might be split across overlapping granules or captured across 20 distinct revisit dates. 
- The hydrator tracks physical `site_key` and centroid coordinates.
- If two candidate tiles fall within a **~200-meter radius** ($0.002^\circ$), only the highest-scoring candidate is kept.
- This guarantees that Top-$K$ results display $K$ **distinct physical geographical locations**, rather than 5 repeated snapshots of the same tarmac.

### 5.3 Automated 3-Index Tactical Spot Description Generator
For every retrieved tile, `generate_tile_description()` evaluates its multi-spectral index profile (`NDVI x NDWI x NDBI`) to synthesize rule-based terrain assessments:
- **Runways / Logistics Yards:** `NDBI >= 0.03` and `NDVI < 0.18` and `NDWI < -0.15`  
  > *"High-density paved industrial or transport corridor (runways, logistics yards, tarmac aprons) with minimal vegetation."*
- **Dense Urban Core:** `NDBI >= 0.04` and `NDVI < 0.24`  
  > *"Dense urban core characterized by concrete infrastructure, built structures, and road networks."*
- **Riparian Corridor / Wetlands:** `NDWI >= 0.05` and `NDVI >= 0.20`  
  > *"Riparian wetland or active riverbank corridor exhibiting rich soil moisture and water-tolerant flora."*
- **Dense Woodland / Forest:** `NDVI >= 0.40` and `NDBI < -0.05`  
  > *"Dense natural woodland with high chlorophyll biomass and zero artificial built-up footprint."*

---

## 6. REST API Endpoints & Request/Response Contracts

The Semantic Retrieval Pipeline exposes two primary endpoints under `/api/v1/search`:

### 6.1 `POST /api/v1/search` (Natural Language Vector Search)
**Request Body (`application/json`):**
```json
{
  "query_text": "airfield runway with visible hangars near river",
  "top_k": 5,
  "analyst_id": "analyst_dgis_01",
  "min_similarity": 0.65,
  "filters": {
    "sensor": "Sentinel-2",
    "start_date": "2023-01-01T00:00:00Z",
    "end_date": "2026-12-31T23:59:59Z",
    "min_quality": 0.80,
    "max_cloud_pct": 15.0,
    "bbox": [72.50, 23.00, 73.00, 23.50]
  }
}
```

**Response (`200 OK`):**
```json
{
  "query_type": "text",
  "query": "airfield runway with visible hangars near river",
  "total_found": 5,
  "filters_applied": {
    "sensor": "Sentinel-2",
    "min_quality": 0.80,
    "max_cloud_pct": 15.0
  },
  "results": [
    {
      "tile_id": "S2B_MSIL2A_20230410_T42QVL_tile_00003",
      "score": 0.884,
      "scene_id": "S2B_MSIL2A_20230410_T42QVL",
      "site_key": "site_23.2154_72.7612_b4c8f1",
      "acquisition_date": "2023-04-10T05:40:12",
      "centroid_lat": 23.2154,
      "centroid_lon": 72.7612,
      "sensor": "Sentinel-2",
      "cloud_pct": 0.012,
      "quality_confidence": 0.988,
      "mean_ndvi": 0.085,
      "mean_ndwi": -0.190,
      "mean_ndbi": 0.115,
      "spot_description": "High-density paved industrial or transport corridor (e.g. runways, logistics yards, tarmac aprons) with minimal vegetation. Spectral signature reflects minimal vegetation (NDVI 0.09), non-water terrestrial surface (NDWI -0.19), urban / built structural materials (NDBI 0.12).",
      "file_path": "data/tiles/western_sector/2023-04-10/tile_00003.tif",
      "thumbnail_url": "/data/tiles/western_sector/2023-04-10/thumb_00003.jpg",
      "geometry_geojson": {
        "type": "Polygon",
        "coordinates": [[[72.75, 23.20], [72.77, 23.20], [72.77, 23.22], [72.75, 23.22], [72.75, 23.20]]]
      }
    }
  ],
  "execution_time_ms": 42.1
}
```

### 6.2 `POST /api/v1/search/image` (Multimodal Image-to-Image Search)
**Request (`multipart/form-data`):**
- `file`: Reference satellite image file (`.jpg`, `.png`, or `.tif`).
- `top_k`: Integer (default `5`).
- `min_similarity`: Float (default `0.65`).
- `sensor`, `start_date`, `end_date`, `min_quality`, `max_cloud_pct`, `aoi_geojson`.

**Response (`200 OK`):** Schema identical to `POST /api/v1/search`.

---

## 7. Operational Guarantees & Analyst Inspection

1. **Sub-Second Latency:** Query embedding and Qdrant graph traversal consistently execute in under 50 ms, ensuring rapid exploration across millions of indexed tiles.
2. **Deep Inspection Modal:** Every retrieved card provides one-click visual preview, coordinate verification, dynamic spectral gauge meters, and direct GeoTIFF download for military GIS workstations (QGIS/ArcGIS).
3. **Audit Logging:** Every query, analyst identity, execution latency, and retrieved tile IDs are logged to the PostgreSQL `search_log` table for defense compliance and intelligence audit trails.
