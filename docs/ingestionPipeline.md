# 🛰️ Canopus Ingestion & Preprocessing Pipeline Specification

This document provides the complete, end-to-end technical specification for the **Canopus Ingestion & Preprocessing Pipeline** (`backend/ingestion/`). It details how raw Earth Observation rasters are validated, aligned across diverse sensors, atmospherically calibrated, partitioned into uniform georeferenced tiles, embedded into high-dimensional vector space, and atomically indexed into spatial and vector databases.

---

## 1. Problem Statement Requirements Addressed by this Pipeline

The Ingestion Pipeline directly resolves the operational demands specified in Ministry of Defence (MoD) / Indian Army Problem Statement **SIH-26227**:

| PS Requirement | Capability Addressed | Pipeline Implementation |
|---|---|---|
| **PS 2.2.1 / 2.2.6** | **Dual Entry Points & Ingestion** | Supports interactive AOI polygon catalog streaming (Copernicus STAC) and 100% offline local GeoTIFF / multi-band folder evaluation ingestion. |
| **PS 2.2.3** | **Tier-1 False-Alarm Suppression** | Eliminates ephemeral atmospheric noise via machine-learning cloud detection (`s2cloudless`), directional cloud-shadow ray-tracing, and mask-aware radiometric percentile normalization. Generates per-pixel georeferenced `_mask.tif` files. |
| **PS 2.2.6** | **Scale & Incremental Ingestion** | Indexes new satellite scenes incrementally into Qdrant (`PointStruct` upsert) and PostgreSQL (`ON CONFLICT (tile_id) DO UPDATE`) without rebuilding existing indices. |
| **PS 2.2.6** | **100% On-Premises Sovereignty** | Completely air-gapped execution using local Docker containers, local model weights (`RemoteCLIP-ViT-B-32.pt`), and local GDAL/PyTorch runtimes with zero external cloud or API dependencies. |
| **PS 2.2.6** | **Geospatial Provenance & Formats** | Strictly preserves native CRS georeferencing (EPSG:4326 via `rasterio.transform`) and standard GeoTIFF formats with LZW compression. |

---

## 2. Raster Fundamentals & Multi-Sensor Input Handling

### 2.1 What is Raster Form in Earth Observation?
In satellite data, imagery is not represented as vector shapes (points/lines/polygons); it is ingested in **raster form**:
* A raster is a regular 2-dimensional grid (matrix) of cells called **pixels**.
* Each pixel corresponds to a precise geographic footprint on the Earth's surface defined by an **Affine Geotransform** (`from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)`).
* In multi-spectral remote sensing, each pixel holds multiple numeric values (Digital Numbers / surface reflectance) across distinct electromagnetic wavelengths (Bands). In memory, this is loaded as a 3D NumPy tensor of shape: `(Bands, Height, Width)`.

### 2.2 Multi-Sensor Compatibility (`backend/ingestion/canvas.py`)
Different satellites capture different band arrangements and radiometric ranges:
- **Sentinel-2 L2A**: Captures 12 spectral bands (including Blue `B02`, Green `B03`, Red `B04`, NIR `B08`, SWIR-1 `B11`, SWIR-2 `B12`, and 10m True Color `TCI`).
- **Landsat 8/9**: Captures Coastal `B1`, Blue `B2`, Green `B3`, Red `B4`, NIR `B5`, SWIR-1 `B6`, SWIR-2 `B7`.
- **Maxar WorldView / Optical Aerial**: High-resolution 3-band True Color RGB or 4-band multispectral optical imagery.

The pipeline achieves **sensor-proof ingestion** via `assemble_canvas_from_file()` and `assemble_mosaicked_canvas_from_stac()`:
1. **Dynamic Band Mapping:** Accepts either multi-band rasters or separate single-band `.tif` files. Maps arbitrary input band configurations into standard semantic channels (`blue`, `green`, `red`, `nir`, `swir`) using automatic sensor detection or user-supplied `custom_band_order`.
2. **On-the-Fly Reprojection:** Satellite scenes often arrive in local UTM projections (e.g. `EPSG:32643`). The canvas dynamically reprojects rasters into **EPSG:4326 (WGS 84)** using `rasterio.vrt.WarpedVRT` with bilinear resampling, guaranteeing sub-pixel spatial alignment across heterogeneous platforms.

---

## 3. Tier-1 False-Alarm Suppression Strategy

A primary failure mode in automated military change detection is false alarms triggered by transient atmospheric interference (clouds, cloud shadows, haze, seasonal illumination changes). The Canopus architecture eliminates these artifacts at the source before tiles enter the archive:

```
[Raw Satellite Bands] ──► [s2cloudless ML Model] ──► [cloud_mask (bool)]
                                                              │
                                                              ▼
[Sun Azimuth / Raycast] ──► [Shadow Adapter] ──────► [shadow_mask (bool)]
                                                              │
                                                              ▼
                                                   [Combined bad_mask]
                                                              │
                     ┌────────────────────────────────────────┴────────────────────────────────────────┐
                     ▼                                                                                 ▼
     [Mask-Aware Percentile Normalization]                                             [Fixed-Scale Reflectance Scaling]
      0.5% - 99.5% on clean ground (~bad_mask)                                              DN / 10,000 -> [0.0, 1.5]
                     │                                                                                 │
                     ▼                                                                                 ▼
           [{tile_id}_thumb.jpg]                                                               [{tile_id}.tif]
     (Human UI & RemoteCLIP Embedding)                                                (Physical Multi-Band GeoTIFF)
```

### 3.1 Machine Learning Cloud Detection (`s2cloudless`)
- Uses the `s2cloudless` gradient-boosted tree model operating over 10 Sentinel-2 bands (including coastal aerosol, water vapor, and cirrus bands).
- Computes per-pixel cloud probabilities $\in [0.0, 1.0]$.
- Applies an operational threshold (default $0.40$) to generate a binary `cloud_mask` (`True = cloud`).

### 3.2 Directional Shadow Ray-Tracing (`backend/ingestion/masking/shadow_adapter.py`)
- Cloud shadows cause sudden drops in surface reflectance, falsely mimicking water bodies or land clearance.
- The pipeline projects shadow vectors along the satellite's solar azimuth and zenith angles, matching dark ground pixels within the projected shadow path to create a binary `shadow_mask`.

### 3.3 Quality Mask Combination & GeoTIFF Export
- Merges masks into a unified boolean matrix:
  ```python
  bad_mask = cloud_mask | shadow_mask
  ```
- For each generated 512×512 tile, the bad mask is written to disk as `{tile_id}_mask.tif` (a 1-band `uint8` GeoTIFF where `0 = clean ground`, `1 = cloud/shadow`).
- Calculates aggregate metrics:
  ```python
  tile_cloud_pct = np.sum(bad_mask) / total_pixels
  quality_confidence = max(0.0, min(1.0, 1.0 - tile_cloud_pct))
  ```

---

## 4. Dual Normalization Architecture

A critical innovation in Canopus is the separation of **Visual Contrast Normalization** from **Physical Reflectance Normalization**:

### 4.1 Adaptive Percentile Normalization (For Visual Thumbnails & CLIP)
* If raw optical bands are saved directly as RGB images, varying sun angles and seasonal atmospheric haze cause dark or washed-out images.
* Function: `normalize_rgb_percentile()` in `backend/ingestion/normalization/percentile_normalization.py`.
* Clips pixel intensities strictly between the **0.5th and 99.5th percentiles of valid, clean ground pixels** (`~bad_mask`).
* Applies gamma adjustment ($\gamma = 1.0$ or $0.85$) and converts the result to 8-bit RGB (`uint8` $[0, 255]$).
* Saved to disk as: `{tile_id}_thumb.jpg`.
* Purpose: Provides clean, high-contrast inputs for human inspection and RemoteCLIP vision encoder extraction.

### 4.2 Fixed-Scale Physical Reflectance (For Multi-Band GeoTIFFs)
* **Crucial Rule:** Adaptive percentile stretching must **NEVER** be applied to the multi-band scientific GeoTIFFs. Doing so changes the pixel numbers arbitrarily based on scene content, destroying radiometric comparability between $T_1$ and $T_2$ acquisitions.
* Function: `to_fixed_reflectance()` in `backend/ingestion/tiler.py`.
* Converts raw Digital Numbers (DN) using standard Sentinel-2 / Landsat fixed scaling:
  ```python
  Reflectance = np.clip(DN / 10000.0, 0.0, 1.5)
  ```
* Saved to disk as: `{tile_id}.tif` (`float32` multi-band GeoTIFF with EPSG:4326 transform).
* Purpose: Preserves genuine physical surface reflectance for mathematical delta comparisons across years.

---

## 5. Equidistant 512×512 Tiling & Spectral Index Extraction

### 5.1 Grid Slicing Algorithm (`backend/ingestion/tiler.py`)
- The reprojected working canvas is sliced into uniform **512×512 pixel patches** using a fixed-stride sliding window (`_get_grid_steps()`):
  ```python
  stride = int(crop_size * (1.0 - overlap_pct))
  ```
- Default ground crop: `512 px` with `10% overlap` (`51 px`), ensuring zero boundary gaps or missing targets along tile edges.
- Each tile receives a deterministic, coordinate-based identifier:
  ```python
  tile_id = f"{clean_scene_id}_tile_{index:05d}"
  site_key = f"site_{round(lat, 4)}_{round(lon, 4)}_{spatial_hash}"
  ```
  *(The `site_key` is identical across multi-temporal acquisitions of the exact same ground location).*

### 5.2 Spectral Indices Computation (Clean-Pixel Masked)
Spectral indices are calculated on the multi-band arrays, **strictly excluding** pixels flagged in `bad_mask` or NoData sentinels (`-9999`, `NaN`, `Inf`):

1. **NDVI (Normalized Difference Vegetation Index):**
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$$
   *Measures vegetation health, canopy density, and agricultural crops.*

2. **NDWI (Normalized Difference Water Index):**
   $$\text{NDWI} = \frac{\text{Green} - \text{NIR}}{\text{Green} + \text{NIR}}$$
   *Delineates open water bodies, rivers, wetlands, and surface moisture.*

3. **NDBI (Normalized Difference Built-Up Index):**
   $$\text{NDBI} = \frac{\text{SWIR} - \text{NIR}}{\text{SWIR} + \text{NIR}}$$
   *Identifies concrete infrastructure, urban density, runways, and bare soil.*

The mean values (`mean_ndvi`, `mean_ndwi`, `mean_ndbi`) are calculated and recorded in database tables and vector payloads.


---

## 6. Sovereign File Storage Hierarchy (`backend/ingestion/storage.py`)

All outputs are saved to local persistent storage in a strict, standardized hierarchy:

```
data/tiles/{region_id}/{YYYY-MM-DD}/
   ├── {tile_id}.tif          # Multi-band fixed-scale reflectance GeoTIFF (EPSG:4326)
   ├── {tile_id}_thumb.jpg    # 8-bit RGB visual thumbnail (512x512 JPEG, quality 95)
   ├── {tile_id}_mask.tif     # 1-band uint8 bad-pixel mask (0=good, 1=bad/cloud/shadow)
   └── manifest.json          # Complete JSON manifest of all tiles, indices, and bounds
```

---

## 7. Dual Database Synchronization (PostGIS + Qdrant)

### 7.1 PostgreSQL / PostGIS Spatial Registration (`backend/ingestion/db_writer.py`)
Metadata and spatial geometries are persisted into PostgreSQL 16:
- **`scenes` Table:** Stores scene identifier, acquisition date, CRS, sensor name, and true bounding footprint polygon (`footprint_geom`).
- **`tiles` Table:** Stores `tile_id`, `scene_id`, `site_key`, `centroid_lat`, `centroid_lon`, `cloud_pct`, `quality_confidence`, `mean_ndvi`, `mean_ndwi`, `mean_ndbi`, `geometry` (PostGIS 4326 polygon), `file_path`, `thumbnail_path`, and `bad_mask_path`.
- **`ingestion_coverage` Table:** Updates the overall region boundary polygon, total tiles ingested, status (`"done"`), and last update timestamp.

### 7.2 RemoteCLIP Embedding & Qdrant Vector Upsert (`backend/services/vector_store.py`)
- Extracts the normalized RGB arrays from each tile candidate.
- Encodes tiles in batches through the fine-tuned **RemoteCLIP ViT-B-32** vision encoder (`encode_image()`), producing a **512-dimensional vector embedding**.
- Upserts points into Qdrant collection `tile_embeddings` (or `maxar_tile_embeddings` for high-res imagery):
  - **Point ID:** Deterministic UUID generated from `tile_id` via `uuid.uuid5(uuid.NAMESPACE_DNS, tile_id)`.
  - **Vector:** 512-dimensional float32 embedding vector (Cosine distance).
  - **Payload:**
    ```json
    {
      "tile_id": "S2A_MSIL2A_20230315_tile_00001",
      "scene_id": "S2A_MSIL2A_20230315_T43REQ",
      "site_key": "site_28.5521_77.1843_a1b2c3d4",
      "region_id": "delhi_ncr_01",
      "sensor": "Sentinel-2",
      "acquisition_date": 1678838400,
      "acquisition_date_iso": "2023-03-15T05:40:12Z",
      "centroid_lat": 28.5521,
      "centroid_lon": 77.1843,
      "cloud_pct": 0.021,
      "quality_confidence": 0.979,
      "mean_ndvi": 0.284,
      "mean_ndwi": -0.142,
      "mean_ndbi": 0.082,
      "source_type": "aoi_search"
    }
    ```

---

## 8. Link to Downstream Change Detection

The files generated during ingestion (`{tile_id}.tif` and `{tile_id}_mask.tif`) directly empower the downstream change detection engine (`backend/services/change_pair.py`):

```python
# Downstream Change Detection Execution
report = resolve_change_pair(
    tile_before_path="data/tiles/delhi/2020-03-15/tile_00001.tif",
    tile_after_path="data/tiles/delhi/2024-03-15/tile_00001.tif",
    mask_before_path="data/tiles/delhi/2020-03-15/tile_00001_mask.tif",
    mask_after_path="data/tiles/delhi/2024-03-15/tile_00001_mask.tif",
    min_usable_fraction=0.30,
    diff_threshold=0.15
)
```

1. **Per-Pixel Mask Combination:** `combined_bad_mask = mask_before | mask_after`.
2. **Valid Ground Gating:** Evaluates the usable ground fraction. If less than 30%, the pair is safely flagged as `insufficient_coverage` rather than raising false alarms.
3. **True Ground Delta:** Physical surface reflectance differences and index shifts (`Delta_NDVI`, `Delta_NDBI`, `Delta_NDWI`) are computed strictly over clean pixels (`~combined_bad_mask`).

---

## 9. Scale, Incremental Ingestion & Operational Sovereignty (PS 2.2.6)

The Ingestion Pipeline fully closes the requirements of PS Section 2.2.6:

1. **Incremental Ingestion Without Index Rebuild:**  
   When a new scene is acquired, only the incoming scene is processed. PostGIS updates via `ON CONFLICT (tile_id) DO UPDATE`, and Qdrant inserts only the newly generated points via `client.upsert()`. Existing tiles, spatial indexes, and vector collections remain completely untouched.  
   👉 **Detailed Procedure:** For the step-by-step operator walkthrough, see [Index-Build & Incremental Ingestion Procedure (`docs/incremental_ingestion_procedure.md`)](incremental_ingestion_procedure.md).

2. **100% On-Premises Air-Gapped Operation:**  
   Requires zero cloud calls or SaaS vector databases. The PostgreSQL, Qdrant, and Python processing containers communicate over a private Docker network with local disk volumes.

3. **Georeferenced Standard Formats:**  
   All output tiles are valid OGC-compliant GeoTIFFs with embedded affine transformations, directly inspectable in military GIS packages such as QGIS or ArcGIS.
