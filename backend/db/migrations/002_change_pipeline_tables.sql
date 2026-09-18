-- ============================================================
-- Migration: 002_change_pipeline_tables.sql
-- Description: Multi-temporal Change Pipeline Persistence Engine
-- Stores complete stage-by-stage audit records:
-- 1. change_runs (Target tile selection & available timeline)
-- 2. change_quality_audits (Quality check engine results per epoch)
-- 3. change_temporal_splits (Temporal splitting & change onset detection)
-- 4. change_clusters (Spatial clusters & semantic transitions)
-- 5. change_candidate_patches (Individual 64x64 candidate patches with PostGIS geometry)
-- ============================================================

CREATE EXTENSION IF NOT EXISTS postgis;

-- 1. CHANGE_RUNS: High-level pipeline execution records
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

-- 2. CHANGE_QUALITY_AUDITS: Per-epoch quality mask engine inspection
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

-- 3. CHANGE_TEMPORAL_SPLITS: Splitting engine & change onset dating
CREATE TABLE IF NOT EXISTS change_temporal_splits (
    split_id                SERIAL PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    method                  TEXT NOT NULL, -- 'prithvi_temporal_step' / 'step_detection'
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

-- 4. CHANGE_CLUSTERS: Spatial cluster semantic typing & physical metrics
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

-- 5. CHANGE_CANDIDATE_PATCHES: Individual 64x64 candidate patches with PostGIS polygon geometry
CREATE TABLE IF NOT EXISTS change_candidate_patches (
    candidate_id            TEXT PRIMARY KEY,
    run_id                  TEXT NOT NULL REFERENCES change_runs(run_id) ON DELETE CASCADE,
    patch_id                INT NOT NULL,
    grid_row                INT NOT NULL,
    grid_col                INT NOT NULL,
    bbox_px                 INT[] NOT NULL, -- [x0, y0, x1, y1]
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
