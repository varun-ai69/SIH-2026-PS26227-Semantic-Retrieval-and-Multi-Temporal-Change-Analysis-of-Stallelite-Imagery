-- schema.sql
-- Final consolidated schema — Semantic Retrieval & Multi-Temporal Change
-- Analysis of Satellite Imagery (PS-227 / SIH-26227)
--
-- Covers every table discussed: scenes, tiles, change_events, clusters,
-- review_items, exports, search_log, ingestion_coverage.
--
-- To auto-load via docker-compose: mount this as
--   /docker-entrypoint-initdb.d/01_schema.sql
-- (runs once, automatically, on first container start — see docker-compose.yml)

CREATE EXTENSION IF NOT EXISTS postgis;

-- ============================================================
-- 1. SCENES — one row per raw satellite download (before tiling)
-- ============================================================
CREATE TABLE IF NOT EXISTS scenes (
  scene_id          TEXT PRIMARY KEY,
  source            TEXT,                          -- 'sentinel2', 'bhoonidhi', etc.
  acquisition_date  TIMESTAMP,                      -- when the satellite captured it
  crs               TEXT,                           -- coordinate system of the raw file
  footprint         GEOMETRY(Polygon, 4326),        -- big polygon, whole-scene coverage
  ingested_at       TIMESTAMP DEFAULT now(),         -- when WE processed it (≠ acquisition_date)
  raw_file_path     TEXT,
  provenance        JSONB                           -- licence, source URL, original product ID
);

-- ============================================================
-- 2. TILES — one row per 512x512 crop, cut from a scene
-- ============================================================
CREATE TABLE IF NOT EXISTS tiles (
  tile_id                TEXT PRIMARY KEY,
  scene_id                TEXT REFERENCES scenes(scene_id),
  site_key                TEXT,                     -- stable ID for "this physical spot",
                                                       -- shared across every date's tile of it
  geometry                GEOMETRY(Polygon, 4326),   -- this tile's own small footprint
  centroid_lat             FLOAT,
  centroid_lon             FLOAT,
  acquisition_date         TIMESTAMP,
  sensor                    TEXT,
  cloud_pct                 FLOAT,                    -- 0-1, from cloud/shadow masking
  quality_confidence          FLOAT,                    -- combined gate (cloud + registration + season)
  cluster_id                     TEXT,                   -- filled by the discovery/clustering job
  file_path                            TEXT,               -- main GeoTIFF (full bit depth/bands)
  thumbnail_path                        TEXT,               -- 8-bit RGB preview JPG
  bad_mask_path                         TEXT,               -- per-pixel bad-pixel/cloud mask GeoTIFF path
  pixel_scale                           TEXT DEFAULT 'reflectance_fixed_10000', -- reflectance scaling standard
  band_order                              JSONB,              -- e.g. ["blue", "green", "red", "nir", "swir"]
  band_stats                               JSONB,              -- per-band min, max, mean summary
  mean_ndvi                                 FLOAT,              -- (NIR - Red) / (NIR + Red)
  mean_ndwi                                  FLOAT,              -- (Green - NIR) / (Green + NIR)
  mean_ndbi                                   FLOAT,              -- (SWIR - NIR) / (SWIR + NIR)
  source_type                                  TEXT DEFAULT 'aoi_search', -- 'aoi_search' or 'organiser_provided'
  mosaicked_scenes                             JSONB,              -- list of contributing scene IDs for mosaicked tiles
  created_at                                    TIMESTAMP DEFAULT now()
);

CREATE INDEX IF NOT EXISTS tiles_geom_idx ON tiles USING GIST (geometry);
CREATE INDEX IF NOT EXISTS tiles_date_idx ON tiles (acquisition_date);
CREATE INDEX IF NOT EXISTS tiles_site_idx ON tiles (site_key);
CREATE INDEX IF NOT EXISTS tiles_cluster_idx ON tiles (cluster_id);
CREATE INDEX IF NOT EXISTS tiles_ndvi_idx ON tiles (mean_ndvi);
CREATE INDEX IF NOT EXISTS tiles_ndbi_idx ON tiles (mean_ndbi);

-- ============================================================
-- 3. CHANGE_EVENTS — output of comparing tile_before vs tile_after
--    at the same site_key
-- ============================================================
CREATE TABLE IF NOT EXISTS change_events (
  event_id                SERIAL PRIMARY KEY,
  site_key                 TEXT,
  tile_before               TEXT REFERENCES tiles(tile_id),
  tile_after                 TEXT REFERENCES tiles(tile_id),
  change_type                 TEXT,      -- construction / clearance / water / road / other
  confidence                   FLOAT,     -- 0-1, model's confidence this is a REAL change
  earliest_visible_date         TIMESTAMP, -- from the backward time-walk
  detected_date                  TIMESTAMP DEFAULT now(),  -- when WE ran this comparison
  model_version                   TEXT,
  quality_flag                     TEXT   -- 'ok' / 'insufficient_coverage' / 'low_confidence'
);

CREATE INDEX IF NOT EXISTS change_events_site_idx ON change_events (site_key);
CREATE INDEX IF NOT EXISTS change_events_date_idx ON change_events (detected_date);
CREATE INDEX IF NOT EXISTS change_events_conf_idx ON change_events (confidence);

-- ============================================================
-- 4. CLUSTERS — groups discovered by the HDBSCAN job over embeddings
-- ============================================================
CREATE TABLE IF NOT EXISTS clusters (
  cluster_id               TEXT PRIMARY KEY,
  label                    TEXT,                 -- optional human-readable name
  representative_tile_id   TEXT,                 -- medoid representative tile
  tile_count               INT DEFAULT 0,
  computed_at              TIMESTAMP DEFAULT now(),
  model_version            TEXT
);

-- ============================================================
-- 5. REVIEW_ITEMS — the analyst queue + audit trail
--    (every status change is a new row's worth of history by design —
--    decided_at is the timestamp that makes this an audit trail)
-- ============================================================
CREATE TABLE IF NOT EXISTS review_items (
  item_id       SERIAL PRIMARY KEY,
  event_id       INT REFERENCES change_events(event_id),
  tile_id         TEXT REFERENCES tiles(tile_id),
  rank_score       FLOAT,
  status            TEXT DEFAULT 'pending',            -- pending / confirmed / rejected
  analyst_id         TEXT DEFAULT 'demo_analyst',       -- no auth yet — placeholder, see note below
  decided_at           TIMESTAMP,
  query_context          JSONB                          -- which search/query produced this item
);

-- ============================================================
-- 6. EXPORTS — provenance-carrying export log
-- ============================================================
CREATE TABLE IF NOT EXISTS exports (
  export_id      SERIAL PRIMARY KEY,
  requested_by     TEXT DEFAULT 'demo_analyst',
  requested_at       TIMESTAMP DEFAULT now(),
  item_ids              INT[],
  format                  TEXT,      -- csv / geojson / pdf
  file_path                 TEXT
);

-- ============================================================
-- 7. SEARCH_LOG — history of semantic_search queries, feeds the
--    reranker's feedback loop later
-- ============================================================
CREATE TABLE IF NOT EXISTS search_log (
  search_id       SERIAL PRIMARY KEY,
  analyst_id        TEXT DEFAULT 'demo_analyst',
  raw_query           TEXT,        -- exactly what the analyst typed
  query_type            TEXT,      -- 'text' or 'image'
  filters                 JSONB,   -- AOI/date/sensor filters applied, null if none
  result_tile_ids           TEXT[],
  searched_at                 TIMESTAMP DEFAULT now()
);

-- ============================================================
-- 8. INGESTION_COVERAGE — frontend "what's embedded vs pending" map
--    (standalone, no hard FK — a project-tracking concept, not lineage)
-- ============================================================
CREATE TABLE IF NOT EXISTS ingestion_coverage (
  region_id       TEXT PRIMARY KEY,
  region_name       TEXT,
  geometry            GEOMETRY(Polygon, 4326),
  status                 TEXT DEFAULT 'pending',    -- pending / in_progress / done
  tile_count                INT DEFAULT 0,
  last_updated                TIMESTAMP DEFAULT now()
);


-- ============================================================
-- 9. CHAT_CONVERSATIONS & CHAT_MESSAGES — ChatGPT/Gemini Style History
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_conversations (
  conversation_id   VARCHAR(64) PRIMARY KEY,
  user_id           VARCHAR(128) NOT NULL,
  title             VARCHAR(255) NOT NULL,
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_conversations_user_updated
  ON chat_conversations(user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
  message_id             SERIAL PRIMARY KEY,
  conversation_id        VARCHAR(64) NOT NULL REFERENCES chat_conversations(conversation_id) ON DELETE CASCADE,
  role                   VARCHAR(20) NOT NULL, -- 'user' or 'assistant'
  content                TEXT NOT NULL,
  attached_image_name    TEXT,
  attached_image_preview TEXT,
  query_context          JSONB,
  results                JSONB,
  created_at             TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_conv_created
  ON chat_messages(conversation_id, created_at ASC);

-- ============================================================
-- 10. CHANGE_RUNS & PIPELINE PERSISTENCE (Stages 1 - 5)
-- ============================================================
CREATE TABLE IF NOT EXISTS change_runs (
    run_id                  TEXT PRIMARY KEY,
    target_tile_id          TEXT REFERENCES tiles(tile_id) ON DELETE SET NULL,
    site_key                TEXT NOT NULL,
    region_id               TEXT,
    geometry                GEOMETRY(Polygon, 4326),
    centroid_lat            FLOAT,
    centroid_lon            FLOAT,
    total_timeline_epochs   INT DEFAULT 0,
    available_dates         JSONB DEFAULT '[]'::jsonb,
    quality_total_seen      INT DEFAULT 0,
    quality_passed_count    INT DEFAULT 0,
    quality_dropped_count   INT DEFAULT 0,
    staging_dir             TEXT,
    manifest_path           TEXT,
    status                  TEXT DEFAULT 'completed',
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_change_runs_site_key ON change_runs(site_key);
CREATE INDEX IF NOT EXISTS idx_change_runs_geom ON change_runs USING GIST(geometry);
CREATE INDEX IF NOT EXISTS idx_change_runs_created ON change_runs(created_at DESC);

CREATE TABLE IF NOT EXISTS change_quality_audits (
    audit_id                SERIAL PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    tile_id                 TEXT,
    acquisition_date        TIMESTAMP WITH TIME ZONE,
    year                    INT,
    cloud_pct               FLOAT,
    bad_pixels              INT,
    total_pixels            INT,
    bad_pixel_pct           FLOAT,
    usable_pct              FLOAT,
    decision                VARCHAR(20) NOT NULL, -- 'STAY' or 'DROPPED'
    drop_reasons            JSONB DEFAULT '[]'::jsonb,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_change_qa_run_id ON change_quality_audits(run_id);
CREATE INDEX IF NOT EXISTS idx_change_qa_tile_id ON change_quality_audits(tile_id);
CREATE INDEX IF NOT EXISTS idx_change_qa_decision ON change_quality_audits(decision);

CREATE TABLE IF NOT EXISTS change_temporal_splits (
    split_id                SERIAL PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    method                  TEXT NOT NULL,
    total_clean_epochs      INT DEFAULT 0,
    optimal_split_index     INT DEFAULT 0,
    step_score              FLOAT,
    split_date_before       TIMESTAMP WITH TIME ZONE,
    split_date_after        TIMESTAMP WITH TIME ZONE,
    onset_bracket           TEXT,
    onset_bracket_days      INT,
    device                  TEXT,
    device_name             TEXT,
    all_split_scores        JSONB DEFAULT '[]'::jsonb,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_change_splits_run_id ON change_temporal_splits(run_id);

CREATE TABLE IF NOT EXISTS change_clusters (
    cluster_db_id           SERIAL PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    cluster_id              INT NOT NULL,
    predicted_type          TEXT NOT NULL,
    cluster_transition      TEXT NOT NULL,
    confidence_pct          FLOAT,
    patch_count             INT DEFAULT 0,
    area_m2                 FLOAT DEFAULT 0.0,
    mean_delta_ndbi         FLOAT,
    mean_delta_ndvi         FLOAT,
    mean_delta_ndwi         FLOAT,
    mean_delta_bsi          FLOAT,
    interpretation          TEXT,
    ai_model                TEXT,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_change_clusters_run_id ON change_clusters(run_id);
CREATE INDEX IF NOT EXISTS idx_change_clusters_type ON change_clusters(predicted_type);

CREATE TABLE IF NOT EXISTS change_candidate_patches (
    candidate_id            TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    patch_id                INT NOT NULL,
    grid_row                INT NOT NULL,
    grid_col                INT NOT NULL,
    bbox_px                 INT[] NOT NULL,
    geometry                GEOMETRY(Polygon, 4326),
    centroid_lat            FLOAT,
    centroid_lon            FLOAT,
    cluster_id              INT,
    distance                FLOAT,
    z_score                 FLOAT,
    predicted_type          TEXT,
    transition_label        TEXT,
    confidence_pct          FLOAT,
    ndvi_t1                 FLOAT,
    ndvi_t2                 FLOAT,
    delta_ndvi              FLOAT,
    ndwi_t1                 FLOAT,
    ndwi_t2                 FLOAT,
    delta_ndwi              FLOAT,
    ndbi_t1                 FLOAT,
    ndbi_t2                 FLOAT,
    delta_ndbi              FLOAT,
    bad_frac_t1             FLOAT,
    bad_frac_t2             FLOAT,
    source_tile_before      TEXT REFERENCES tiles(tile_id) ON DELETE SET NULL,
    source_tile_after       TEXT REFERENCES tiles(tile_id) ON DELETE SET NULL,
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_change_patches_run_id ON change_candidate_patches(run_id);
CREATE INDEX IF NOT EXISTS idx_change_patches_cluster ON change_candidate_patches(cluster_id);
CREATE INDEX IF NOT EXISTS idx_change_patches_geom ON change_candidate_patches USING GIST(geometry);

-- ============================================================
-- 10. ANALYST WORKFLOW & PROVENANCE TABLES
-- ============================================================
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- 10.1 ANALYST_DECISIONS: Immutable append-only audit trail of analyst change decisions
CREATE TABLE IF NOT EXISTS analyst_decisions (
    decision_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id          TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    candidate_id    TEXT NOT NULL,
    analyst_id      VARCHAR(100) NOT NULL DEFAULT 'ANALYST-DEF-01',
    decision        VARCHAR(30) NOT NULL, -- 'confirmed', 'rejected', 'unsure'
    note            TEXT DEFAULT '',
    tags            JSONB DEFAULT '[]'::jsonb,
    decided_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_analyst_decisions_run_cand ON analyst_decisions (run_id, candidate_id);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_decided_at ON analyst_decisions (decided_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_analyst ON analyst_decisions (analyst_id);
CREATE INDEX IF NOT EXISTS idx_analyst_decisions_decision ON analyst_decisions (decision);

-- 10.2 SEARCH_FEEDBACK: Relevance feedback for Semantic Retrieval
CREATE TABLE IF NOT EXISTS search_feedback (
    feedback_id     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    analyst_id      VARCHAR(100) NOT NULL DEFAULT 'ANALYST-DEF-01',
    query_text      TEXT NOT NULL,
    tile_id         TEXT REFERENCES tiles(tile_id) ON DELETE CASCADE,
    relevant        BOOLEAN NOT NULL, -- true = Relevant / Hit, false = Irrelevant / False Alarm
    relevance_score FLOAT DEFAULT 1.0,
    tag             VARCHAR(100) DEFAULT '',
    note            TEXT DEFAULT '',
    recorded_at     TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_search_feedback_query ON search_feedback (query_text, recorded_at DESC);
CREATE INDEX IF NOT EXISTS idx_search_feedback_tile ON search_feedback (tile_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_analyst ON search_feedback (analyst_id);
CREATE INDEX IF NOT EXISTS idx_search_feedback_recorded ON search_feedback (recorded_at DESC);
