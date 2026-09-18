# 🌐 Canopus REST API Contract Specification

This document provides the verified API specification for the Canopus backend service (`/api/v1`), detailing active endpoints, HTTP methods, request bodies, query parameters, response structures, and status codes.

---

## 1. System Health & Application Web Endpoints

### `GET /health`
Returns the operational health status of the FastAPI backend service.

- **Request**: None
- **Response**: `200 OK`
```json
{
  "status": "ok"
}
```

### Application Frontend & Documentation Routes
The backend serves the web user interfaces and interactive OpenAPI documentation directly:
- `GET /`: Map Explorer & AOI Ingestion Workspace (`frontend/index.html`)
- `GET /retrieval` or `/retrieval.html`: Multimodal Semantic Retrieval & Persistent Chat Interface
- `GET /clustering` or `/clustering.html`: HDBSCAN Clustering & Medoid Discovery Workbench
- `GET /review` or `/review.html`: Analyst Verification & Quality Audit Feed
- `GET /docs`: Interactive Swagger / OpenAPI REST documentation
- Static Mounts: `/data` (tile storage, thumbnails, GeoTIFFs, manifests) and `/frontend` (styles, scripts, assets)

---

## 2. Geographic Coverage (`/api/v1/coverage`)

### `GET /api/v1/coverage`
Fetches a GeoJSON `FeatureCollection` of all ingested geographic regions currently cataloged in the spatial database (`ingestion_coverage` table), along with overall archive counts.

- **Request**: None
- **Response**: `200 OK`
```json
{
  "type": "FeatureCollection",
  "total_regions": 4,
  "total_tiles": 348,
  "total_scenes": 12,
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [77.10, 28.50],
            [77.30, 28.50],
            [77.30, 28.70],
            [77.10, 28.70],
            [77.10, 28.50]
          ]
        ]
      },
      "properties": {
        "region_id": "region_28.60_77.20",
        "region_name": "Delhi_NCR_Sector",
        "status": "done",
        "tile_count": 16,
        "last_updated": "2026-09-12T18:30:00Z"
      }
    }
  ]
}
```

---

## 3. Ingestion & Preprocessing Pipeline (`/api/v1/ingest`)

### `POST /api/v1/ingest/aoi`
Triggers multi-temporal ingestion for a user-drawn AOI polygon.
- For **Sentinel-2** (`sensor: "sentinel2"`), queries Copernicus STAC, builds WarpedVRT multi-spectral canvases, slices 512×512 tiles, and indexes to Postgres & Qdrant synchronously.
- For **Maxar** (`sensor: "maxar"`), automatically delegates to a background thread and returns an async `job_id` immediately.

- **Request Body**: `application/json`
```json
{
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [
      [
        [77.10, 28.50],
        [77.30, 28.50],
        [77.30, 28.70],
        [77.10, 28.70],
        [77.10, 28.50]
      ]
    ]
  },
  "sensor": "sentinel2",
  "region_id": "delhi_ncr_01",
  "region_name": "Delhi NCR Sector",
  "date_from": "2023-01-01",
  "date_to": "2024-01-01",
  "years_timeline": null,
  "num_time_buckets": 2,
  "max_cloud_cover": 20.0,
  "ground_crop_size": 512,
  "overlap_pct": 0.10,
  "populate_db": true
}
```

- **Response (Sentinel-2 Synchronous)**: `200 OK`
```json
{
  "status": "success",
  "region_id": "delhi_ncr_01",
  "source_type": "sentinel2_stac",
  "total_tiles_generated": 16,
  "scenes_processed": [
    {
      "scene_id": "S2A_MSIL2A_20230315_T43REQ",
      "time_bucket": "2023-01-01_to_2023-07-02",
      "acquisition_date": "2023-03-15",
      "tiles_count": 8,
      "cloud_pct": 3.8,
      "manifest_path": "data/tiles/delhi_ncr_01/manifest.json"
    }
  ],
  "elapsed_seconds": 14.2,
  "is_offline": false
}
```

- **Response (Maxar Background Job Initiated)**: `200 OK`
```json
{
  "status": "queued",
  "job_id": "3a7b9c1d2e4f",
  "region_id": "delhi_ncr_01",
  "message": "Maxar ingestion started in background. Poll /api/v1/ingest/job/3a7b9c1d2e4f for status."
}
```

---

### `POST /api/v1/ingest/file`
Ingests an offline local GeoTIFF raster file or a list of band files directly with custom band mappings and georeferencing.

- **Request Body**: `application/json`
```json
{
  "file_path": "/app/data/sample_scene.tif",
  "file_paths": null,
  "sensor": "auto",
  "region_id": "local_evaluation_sector",
  "custom_band_order": ["blue", "green", "red", "nir", "swir"],
  "acquisition_date": "2024-05-10",
  "ground_crop_size": 512,
  "overlap_pct": 0.10,
  "populate_db": true
}
```
- **Response**: `200 OK` (Schema identical to Sentinel-2 AOI pipeline result).

---

### `POST /api/v1/ingest/upload`
Uploads local GeoTIFF file(s) via multipart form data, applies 512×512 tiling, spectral index calculation, and database indexing.

- **Request**: `multipart/form-data`
  - `files` / `file`: File upload stream
  - `sensor`: Sensor convention string (e.g. `"auto"`, `"sentinel2"`, `"landsat"`, default: `"auto"`)
  - `region_id`: Optional custom region identifier
  - `custom_band_order`: Optional comma-delimited band order (e.g. `"blue,green,red,nir,swir"`)
  - `acquisition_date`: Optional ISO date (`YYYY-MM-DD`)
- **Response**: `200 OK`
```json
{
  "status": "success",
  "region_id": "upload_sample_scene",
  "source_type": "file_upload",
  "files_uploaded": 1,
  "total_tiles_generated": 8,
  "elapsed_seconds": 4.8
}
```

---

### `POST /api/v1/ingest/maxar`
Directly triggers an asynchronous Maxar WorldView / Wayback WMTS high-resolution ingestion job across specified historical epochs. Returns `job_id` instantly to avoid browser timeouts.

- **Request Body**: `application/json`
```json
{
  "geojson_polygon": {
    "type": "Polygon",
    "coordinates": [
      [
        [-76.32, 36.91],
        [-76.28, 36.91],
        [-76.28, 36.95],
        [-76.32, 36.95],
        [-76.32, 36.91]
      ]
    ]
  },
  "region_id": "maxar_norfolk_naval",
  "region_name": "Norfolk Naval Base",
  "years": [2020, 2026],
  "zoom_level": 16,
  "overlap_pct": 0.10,
  "populate_db": true
}
```
- **Response**: `200 OK`
```json
{
  "status": "queued",
  "job_id": "b3c4d5e6f7a8",
  "region_id": "maxar_norfolk_naval",
  "message": "Maxar ingestion started in background. Poll /api/v1/ingest/job/b3c4d5e6f7a8 for status."
}
```

---

### `GET /api/v1/ingest/job/{job_id}`
Polls real-time progress and completion status of an asynchronous background Maxar ingestion job.

- **Path Parameters**:
  - `job_id`: Job UUID string
- **Response**: `200 OK`
```json
{
  "job_id": "b3c4d5e6f7a8",
  "status": "done",
  "progress": 100,
  "message": "Done! 54 tiles across 2 epochs in 32.4s",
  "region_id": "maxar_norfolk_naval",
  "created_at": "2026-09-13T14:10:00.000Z",
  "result": {
    "status": "success",
    "region_id": "maxar_norfolk_naval",
    "sensor": "Maxar WorldView / Wayback",
    "epochs_processed": [2020, 2026],
    "total_tiles_generated": 54,
    "tiles_upserted_postgres": 54,
    "vectors_upserted_qdrant": 54,
    "qdrant_collection": "maxar_tile_embeddings",
    "elapsed_seconds": 32.4,
    "errors": []
  }
}
```

---

## 4. Tile Catalog & Archive Statistics (`/api/v1/archive`)

### `GET /api/v1/archive/stats`
Returns system-wide operational summary statistics and cataloged geographic regions.

- **Response**: `200 OK`
```json
{
  "total_tiles": 348,
  "total_scenes": 12,
  "total_regions": 4,
  "regions": [
    {
      "region_id": "region_28.60_77.20",
      "region_name": "Delhi_NCR_Sector",
      "tile_count": 16,
      "status": "done",
      "last_updated": "2026-09-12T18:30:00Z"
    }
  ]
}
```

---

### `GET /api/v1/archive/tiles`
Queries paginated catalog of tiles with multi-spectral properties and spatial footprints.

- **Query Parameters**:
  - `region_id` (optional): Filter by region string (or `"all"`)
  - `limit` (optional, default: `100`, min: 1, max: 500): Max results to return
  - `offset` (optional, default: `0`, min: 0): Result offset for pagination
- **Response**: `200 OK`
```json
{
  "total": 348,
  "count": 100,
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      },
      "properties": {
        "tile_id": "region_28.60_77.20_20230315_tile_00001",
        "scene_id": "S2A_MSIL2A_20230315_T43REQ",
        "site_key": "delhi_ncr",
        "centroid_lat": 28.5521,
        "centroid_lon": 77.1843,
        "cloud_pct": 2.1,
        "quality_confidence": 0.98,
        "mean_ndvi": 0.284,
        "mean_ndwi": -0.142,
        "mean_ndbi": 0.082,
        "thumbnail_path": "data/tiles/delhi_ncr_01/2023-03-15/thumb_00001.jpg",
        "source_type": "sentinel2_stac"
      }
    }
  ]
}
```

---

## 5. Multimodal Semantic Retrieval & Vector Search (`/api/v1/search`)

### `POST /api/v1/search`
Performs zero-shot natural language vector search against tile embeddings using RemoteCLIP ViT-B-32 with compound metadata and spatial filters.

- **Request Body**: `application/json`
```json
{
  "query_text": "airstrip runway with visible hangars near water",
  "top_k": 5,
  "analyst_id": "demo_analyst",
  "min_similarity": 0.65,
  "filters": {
    "sensor": "Sentinel-2",
    "start_date": "2020-01-01T00:00:00Z",
    "end_date": "2026-12-31T23:59:59Z",
    "min_quality": 0.80,
    "max_cloud_pct": 20.0,
    "aoi_polygon": null,
    "bbox": [72.50, 23.00, 73.00, 23.50]
  }
}
```
- **Response**: `200 OK`
```json
{
  "query_type": "text",
  "query": "airstrip runway with visible hangars near water",
  "total_found": 5,
  "filters_applied": {
    "sensor": "Sentinel-2",
    "min_quality": 0.80,
    "max_cloud_pct": 20.0
  },
  "results": [
    {
      "tile_id": "region_23.20_72.75_20230410_tile_00003",
      "score": 0.884,
      "scene_id": "S2B_MSIL2A_20230410_T42QVL",
      "site_key": "western_airfield",
      "acquisition_date": "2023-04-10T05:40:12Z",
      "centroid_lat": 23.2154,
      "centroid_lon": 72.7612,
      "sensor": "Sentinel-2",
      "cloud_pct": 1.2,
      "quality_confidence": 0.98,
      "mean_ndvi": 0.085,
      "mean_ndwi": -0.190,
      "mean_ndbi": 0.115,
      "spot_description": "High-density paved industrial or transport corridor (e.g. runways, logistics yards, tarmac aprons) with minimal vegetation. Spectral signature reflects minimal vegetation (NDVI 0.09), non-water terrestrial surface (NDWI -0.19), urban / built structural materials (NDBI 0.12).",
      "file_path": "data/tiles/western_airfield/2023-04-10/tile_00003.tif",
      "thumbnail_url": "/data/tiles/western_airfield/2023-04-10/thumb_00003.jpg",
      "thumbnail_path": "data/tiles/western_airfield/2023-04-10/thumb_00003.jpg",
      "geometry_geojson": {
        "type": "Polygon",
        "coordinates": [...]
      }
    }
  ],
  "execution_time_ms": 42.1
}
```

---

### `POST /api/v1/search/image`
Uploads a reference satellite image to perform visual similarity search (image-to-image retrieval) across the vector archive.

- **Request**: `multipart/form-data`
  - `file`: Image file (`.jpg`, `.png`, or `.tif`) — **Required**
  - `top_k`: Integer (default: `5`, 1 to 100)
  - `sensor`: Optional string filter (e.g. `"Sentinel-2"`, `"Maxar"`)
  - `start_date`: Optional ISO start date (`YYYY-MM-DD`)
  - `end_date`: Optional ISO end date (`YYYY-MM-DD`)
  - `min_quality`: Float (default: `0.0`, 0.0 to 1.0)
  - `max_cloud_pct`: Float (default: `100.0`, 0.0 to 100.0)
  - `min_similarity`: Float (default: `0.65`, 0.0 to 1.0)
  - `aoi_geojson`: Optional GeoJSON polygon string for spatial pre-filtering
  - `analyst_id`: String (default: `"demo_analyst"`)
- **Response**: `200 OK` (Schema identical to `POST /api/v1/search`).

---

## 6. Unsupervised Clustering & Similarity Discovery (`/api/v1`)

### `GET /api/v1/discover/{tile_id}`
Given any existing `tile_id`, discovers visually and semantically similar tiles ("digital twins") across the entire archive using its precomputed stored vector. Zero prompt, zero new embedding computation.

- **Path Parameters**:
  - `tile_id`: Database tile identifier string
- **Query Parameters**:
  - `top_k` (default: `10`, min: 1, max: 50): Number of top matches to return
  - `min_similarity` (default: `0.40`, min: 0.0, max: 1.0): Minimum cosine similarity cutoff
- **Response**: `200 OK` (Array of `SearchResultItem` objects).

---

### `GET /api/v1/clusters`
Returns all discovered terrain and land-use clusters generated by unsupervised density partitioning (HDBSCAN), including medoids and spectral signatures.

- **Response**: `200 OK`
```json
[
  {
    "cluster_id": "cluster_20260912223054_10",
    "label": "Cluster 10 — Open Arid & Transition Terrain (NDVI 0.02)",
    "representative_tile_id": "maxar_region_36.92_-76.25_2026_9_3",
    "tile_count": 54,
    "computed_at": "2026-09-12T22:30:54Z",
    "model_version": "RemoteCLIP-ViT-B-32",
    "thumbnail_url": "/data/tiles/region_36.92_-76.25/2026-08-05/thumb.jpg",
    "centroid_lat": 36.9349,
    "centroid_lon": -76.2542,
    "mean_ndvi": 0.0019,
    "sensor": "Maxar WorldView / Wayback"
  }
]
```

---

### `GET /api/v1/clusters/{cluster_id}/tiles`
Retrieves member tiles belonging to a specific cluster ID.

- **Path Parameters**:
  - `cluster_id`: Cluster identifier string
- **Query Parameters**:
  - `limit` (default: `60`, min: 1, max: 120): Maximum member tiles to return
- **Response**: `200 OK` (Array of `SearchResultItem` objects).

---

### `POST /api/v1/clusters/recompute`
Triggers batch HDBSCAN/KMeans clustering over all vector embeddings currently present in Qdrant, refreshing Postgres tables and Qdrant payloads.

- **Response**: `200 OK`
```json
{
  "status": "success",
  "clusters_count": 10,
  "total_points": 348,
  "elapsed_seconds": 2.14
}
```

---

## 7. Persistent Chat System (`/api/v1/chat`)

### `GET /api/v1/chat/conversations`
Lists all conversations associated with a specific user ID, sorted by most recently active.

- **Query Parameters**:
  - `user_id` (required): Unique user session identifier (e.g., `default_analyst`)
- **Response**: `200 OK`
```json
[
  {
    "conversation_id": "conv_9f3b12a8382a",
    "user_id": "default_analyst",
    "title": "Airstrip search near coastal waters",
    "message_count": 4,
    "created_at": "2026-09-13T03:15:00Z",
    "updated_at": "2026-09-13T03:22:10Z"
  }
]
```

---

### `POST /api/v1/chat/conversations`
Creates a new persistent conversation session for the specified user.

- **Request Body**: `application/json`
```json
{
  "user_id": "default_analyst",
  "title": "New Tactical Search",
  "conversation_id": null
}
```
- **Response**: `201 Created`
```json
{
  "conversation_id": "conv_9f3b12a8382a",
  "user_id": "default_analyst",
  "title": "New Tactical Search",
  "created_at": "2026-09-13T03:15:00Z",
  "updated_at": "2026-09-13T03:15:00Z",
  "message_count": 0
}
```

---

### `GET /api/v1/chat/conversations/{conversation_id}`
Fetches full message history, query context, attached images, and search results for a specific conversation with strict user isolation.

- **Path Parameters**:
  - `conversation_id`: Unique conversation identifier
- **Query Parameters**:
  - `user_id` (required): Requesting user identifier
- **Response**: `200 OK`
```json
{
  "conversation_id": "conv_9f3b12a8382a",
  "user_id": "default_analyst",
  "title": "Airstrip search near coastal waters",
  "created_at": "2026-09-13T03:15:00Z",
  "updated_at": "2026-09-13T03:22:10Z",
  "messages": [
    {
      "message_id": 1,
      "conversation_id": "conv_9f3b12a8382a",
      "role": "user",
      "content": "find runways and airstrips",
      "attached_image_name": null,
      "attached_image_preview": null,
      "query_context": null,
      "results": null,
      "created_at": "2026-09-13T03:15:10Z"
    },
    {
      "message_id": 2,
      "conversation_id": "conv_9f3b12a8382a",
      "role": "assistant",
      "content": "Retrieved 5 candidate satellite tiles matching your query.",
      "attached_image_name": null,
      "attached_image_preview": null,
      "query_context": {
        "top_k": 5,
        "min_similarity": 0.65
      },
      "results": [ ... ],
      "created_at": "2026-09-13T03:15:12Z"
    }
  ]
}
```

---

### `PATCH /api/v1/chat/conversations/{conversation_id}`
Renames the title of an existing conversation thread.

- **Path Parameters**:
  - `conversation_id`: Unique conversation identifier
- **Query Parameters**:
  - `user_id` (required): User identity
- **Request Body**: `application/json`
```json
{
  "title": "Naval Base & Anchorage Analysis"
}
```
- **Response**: `200 OK`

---

### `DELETE /api/v1/chat/conversations/{conversation_id}`
Permanently deletes a conversation and cascades deletion to all associated messages.

- **Path Parameters**:
  - `conversation_id`: Unique conversation identifier
- **Query Parameters**:
  - `user_id` (required): User identity
- **Response**: `200 OK`
```json
{
  "status": "deleted",
  "conversation_id": "conv_9f3b12a8382a"
}
```

---

### `POST /api/v1/chat/conversations/{conversation_id}/messages`
Appends one or more messages (user prompt or assistant search response) to the conversation history and automatically updates the conversation's timestamp.

- **Path Parameters**:
  - `conversation_id`: Unique conversation identifier
- **Query Parameters**:
  - `user_id` (required): User identity
- **Request Body**: `application/json` (Single message object or JSON array of message objects)
```json
{
  "role": "user",
  "content": "show airstrips near coastal waters",
  "attached_image_name": null,
  "attached_image_preview": null,
  "query_context": {
    "top_k": 5,
    "min_similarity": 0.65
  },
  "results": null
}
```
- **Response**: `200 OK`
```json
{
  "status": "success",
  "conversation_id": "conv_9f3b12a8382a",
  "title": "show airstrips near coastal waters",
  "inserted_count": 1,
  "messages": [
    {
      "message_id": 3,
      "conversation_id": "conv_9f3b12a8382a",
      "role": "user",
      "content": "show airstrips near coastal waters",
      "attached_image_name": null,
      "attached_image_preview": null,
      "query_context": {
        "top_k": 5,
        "min_similarity": 0.65
      },
      "results": null,
      "created_at": "2026-09-13T03:25:00Z"
    }
  ]
}
```
