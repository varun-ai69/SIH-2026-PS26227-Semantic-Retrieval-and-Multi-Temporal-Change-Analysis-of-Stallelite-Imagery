# 📁 Canopus Repository Directory Structure

This document outlines the organization and layout of the Canopus codebase, detailing key directories, services, and operational components.

```
.
├── docker-compose.yml                      # Top-level Docker Compose configuration (single-command launch)
├── requirements.txt                        # Python production & ML dependencies
├── .env.example                            # Template for environment variables and service ports
├── .dockerignore                           # Build exclusions for container optimization
├── .gitignore                              # Git tracking exclusions (imagery, weights, virtual environments)
├── README.md                               # Main project overview, features, and quickstart
│
├── backend/                                # Core FastAPI backend, ingestion pipeline & AI services
│   ├── api/                                # REST API layer
│   │   ├── main.py                         # FastAPI application entrypoint, CORS & static file mounting
│   │   └── routers/                        # Modular API route controllers
│   │       ├── archive.py                  # Tile catalog, scenes, and archive statistics endpoints
│   │       ├── change.py                   # Multi-temporal change detection & delta analysis
│   │       ├── chat.py                     # Persistent ChatGPT/Gemini-style chat session management
│   │       ├── coverage.py                 # GeoJSON coverage tracking for ingested sectors
│   │       ├── discovery.py                # HDBSCAN clustering & cross-site discovery endpoints
│   │       ├── ingest.py                   # AOI polygon and local GeoTIFF ingestion triggers
│   │       └── search.py                   # Multimodal text & image semantic vector search
│   │
│   ├── db/                                 # PostgreSQL / PostGIS database schemas & models
│   │   ├── schema.sql                      # Spatial DDL, tables (tiles, clusters, chat, scenes) & indexes
│   │   ├── seed.sql                        # Initial seed data for development
│   │   └── models.py                       # SQLAlchemy / GeoAlchemy2 ORM models
│   │
│   ├── ingestion/                          # Preprocessing & Tiling Engine (Phase 1)
│   │   ├── canvas.py                       # 5-band canvas reprojection (WarpedVRT) & TCI streaming
│   │   ├── db_writer.py                    # Atomic database persistence & automated schema migrations
│   │   ├── input_validator.py              # AOI polygon topology & GeoTIFF coordinate validation
│   │   ├── pipeline.py                     # Unified Phase 1 ingestion orchestrator
│   │   ├── stac_search.py                  # Multi-temporal STAC catalog querying & cloud filtering
│   │   ├── storage.py                      # GeoTIFF (.tif), preview (.jpg) & manifest disk writer
│   │   └── tiler.py                        # Equidistant 512x512 slicing & spectral index calculation
│   │
│   ├── maxar_ingestion/                    # High-Resolution Optical Ingestion Pipeline
│   │   ├── fetcher.py                      # Maxar WorldView / Wayback imagery fetcher
│   │   ├── pipeline.py                     # High-resolution tile preprocessing & normalization
│   │   └── tiler.py                        # High-resolution spatial tiler
│   │
│   ├── services/                           # AI / ML & Vector Store Integration (Phase 2 & Phase 3)
│   │   ├── change_pair.py                  # Bi-temporal image pair alignment & delta calculation
│   │   ├── discovery.py                    # Similarity graph discovery & cross-cluster exploration
│   │   ├── encoder.py                      # RemoteCLIP ViT-B-32 vision-language embedding service (512-D)
│   │   ├── vector_search.py                # Qdrant cosine vector matching, filters & hydration
│   │   └── vector_store.py                 # Qdrant vector database client & collection management
│   │
│   ├── jobs/                               # Standalone & Batch Processing Jobs
│   │   └── run_clustering.py               # Unsupervised HDBSCAN clustering job over tile embeddings
│   │
│   ├── cloud_removal/                      # Cloud removal and reconstruction algorithms
│   └── preview/                            # GeoTIFF metadata readers & RGB visualizer
│
├── frontend/                               # Glassmorphic Aerospace Web User Interface
│   ├── index.html                          # Map Explorer & AOI Ingestion Workspace (/)
│   ├── retrieval.html                      # Multimodal Semantic Retrieval & Chat Interface (/retrieval.html)
│   ├── change.html                         # Multi-Temporal Change Detection Workspace (/change.html)
│   ├── clustering.html                     # HDBSCAN Clustering & Discovery Workbench (/clustering.html)
│   ├── review.html                         # Analyst Verification & Audit Tool (/review.html)
│   ├── app.js                              # Client-side map controllers, API clients & chat session logic
│   └── index.css                           # Glassmorphic dark aerospace UI design system
│
├── infra/                                  # Containerization & Deployment
│   ├── docker-compose.yml                  # Infrastructure-level Compose configuration
│   └── docker/
│       ├── Dockerfile                      # Debian/Python 3.11 container with GDAL, PyTorch, LibGEOS
│       └── entrypoint.sh                   # Startup health-check, migration runner & Uvicorn launcher
│
├── models/                                 # Pre-trained deep learning checkpoints
│   └── retrieval/
│       └── RemoteCLIP-ViT-B-32.pt          # Fine-tuned RemoteCLIP ViT-B-32 model weights (~605 MB)
│
├── data/                                   # Local operational storage (mounted as Docker volume)
│   ├── tiles/                              # Partitioned 512x512 GeoTIFFs, thumbnails & manifests
│   └── custom_aoi.geojson                  # Saved geographic boundaries and AOIs
│
├── docs/                                   # Project Documentation & Architecture Records
│   ├── api_contract.md                     # Complete REST API route specifications and schemas
│   ├── repository_structure.md             # Detailed directory structure (this file)
│   └── qgis_visual_inspection.md           # Verification guide for QGIS raster layer inspection
│
└── tests/                                  # Automated Unit & Integration Test Suite
    ├── test_phase1_0_input_handling.py     # Polygon & GeoTIFF input validation tests
    ├── test_phase1_1_stac_search.py        # STAC search & temporal bucketing tests
    ├── test_phase1_2_canvas.py             # Canvas assembly & reprojection tests
    ├── test_phase1_3_quality_masking.py    # s2cloudless cloud masking tests
    ├── test_phase1_4_5_tiling_storage.py   # 512x512 equidistant tiling & storage tests
    ├── test_phase1_8_encoder_vectorstore.py# RemoteCLIP embedding & Qdrant vector tests
    ├── test_phase1_e2e_pipeline.py         # End-to-end ingestion pipeline tests
    ├── test_phase2_retrieval.py            # Multimodal semantic retrieval tests
    └── test_bugfixes.py                    # Regression tests for system fixes
```
