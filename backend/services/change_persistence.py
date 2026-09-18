"""
backend/services/change_persistence.py
======================================
PostgreSQL / PostGIS Persistence Engine for Multi-Temporal Change Detection

Stores stage-by-stage pipeline execution:
  1. Target tile selection, site footprint & available timeline (change_runs)
  2. Quality mask engine audit per epoch: STAY vs DROPPED (change_quality_audits)
  3. Temporal splitting & change onset dating (change_temporal_splits)
  4. Spatial clusters & semantic typing with physical metrics (change_clusters)
  5. Individual 64x64 candidate patches with PostGIS polygon geometry (change_candidate_patches)

Provides GeoJSON export compatible with standard GIS tools (QGIS, Leaflet).
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

import psycopg2.extras
from backend.ingestion.db_writer import get_pg_connection

logger = logging.getLogger("change_persistence")


def compute_patch_polygon_wkt(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    bbox_px: List[int],
    tile_size: int = 512
) -> Tuple[str, float, float]:
    """
    Computes real-world WGS84 coordinates and PostGIS POLYGON WKT
    for a 64x64 patch bounding box [x0, y0, x1, y1] within a 512x512 tile.
    
    Returns:
        (wkt_polygon, centroid_lat, centroid_lon)
    """
    x0, y0, x1, y1 = bbox_px
    fx0 = float(x0) / float(tile_size)
    fx1 = float(x1) / float(tile_size)
    fy0 = float(y0) / float(tile_size)
    fy1 = float(y1) / float(tile_size)

    # Longitude increases left to right (West -> East)
    lon0 = min_lon + (max_lon - min_lon) * fx0
    lon1 = min_lon + (max_lon - min_lon) * fx1

    # Latitude decreases top to bottom (North -> South)
    lat_north = max_lat - (max_lat - min_lat) * fy0
    lat_south = max_lat - (max_lat - min_lat) * fy1

    # Clockwise polygon: top-left -> top-right -> bottom-right -> bottom-left -> top-left
    wkt = f"POLYGON(({lon0} {lat_north}, {lon1} {lat_north}, {lon1} {lat_south}, {lon0} {lat_south}, {lon0} {lat_north}))"
    c_lat = (lat_north + lat_south) / 2.0
    c_lon = (lon0 + lon1) / 2.0

    return wkt, c_lat, c_lon


def save_change_pipeline_run(
    target_tile_id: str,
    site_key: str,
    staging_dir: str,
    manifest_path: str,
    quality_audit: Dict[str, Any],
    splitting_audit: Dict[str, Any],
    patch_change_analysis: Optional[Dict[str, Any]],
    all_sibling_records: List[Dict[str, Any]],
    staged_epochs: List[Dict[str, Any]],
    spectral_deltas: Optional[Dict[str, Any]] = None
) -> str:
    """
    Transactionally persists all 5 stages of the change analysis pipeline
    into PostgreSQL / PostGIS.
    """
    run_id = f"run_{site_key}_{int(time.time())}"
    conn = get_pg_connection()

    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Retrieve target tile geometry and bounds
            cur.execute("""
                SELECT 
                    t.tile_id,
                    t.site_key,
                    t.scene_id,
                    t.file_path,
                    t.centroid_lat,
                    t.centroid_lon,
                    ST_AsText(t.geometry) as geom_wkt,
                    ST_XMin(t.geometry) as min_lon,
                    ST_YMin(t.geometry) as min_lat,
                    ST_XMax(t.geometry) as max_lon,
                    ST_YMax(t.geometry) as max_lat
                FROM tiles t
                WHERE t.tile_id = %s
                LIMIT 1;
            """, (target_tile_id,))
            target_tile = cur.fetchone()

            if not target_tile:
                logger.warning(f"Target tile {target_tile_id} not found; querying by site_key {site_key}")
                cur.execute("""
                    SELECT 
                        t.tile_id,
                        t.site_key,
                        t.scene_id,
                        t.file_path,
                        t.centroid_lat,
                        t.centroid_lon,
                        ST_AsText(t.geometry) as geom_wkt,
                        ST_XMin(t.geometry) as min_lon,
                        ST_YMin(t.geometry) as min_lat,
                        ST_XMax(t.geometry) as max_lon,
                        ST_YMax(t.geometry) as max_lat
                    FROM tiles t
                    WHERE t.site_key = %s
                    ORDER BY t.acquisition_date DESC
                    LIMIT 1;
                """, (site_key,))
                target_tile = cur.fetchone()

            min_lon = float(target_tile["min_lon"]) if target_tile and target_tile.get("min_lon") is not None else 0.0
            min_lat = float(target_tile["min_lat"]) if target_tile and target_tile.get("min_lat") is not None else 0.0
            max_lon = float(target_tile["max_lon"]) if target_tile and target_tile.get("max_lon") is not None else 0.0
            max_lat = float(target_tile["max_lat"]) if target_tile and target_tile.get("max_lat") is not None else 0.0
            c_lat = float(target_tile["centroid_lat"]) if target_tile and target_tile.get("centroid_lat") is not None else 0.0
            c_lon = float(target_tile["centroid_lon"]) if target_tile and target_tile.get("centroid_lon") is not None else 0.0
            geom_wkt = target_tile.get("geom_wkt") if target_tile else None
            
            region_id = None
            if target_tile and target_tile.get("file_path"):
                fp = str(target_tile["file_path"]).replace("\\", "/")
                if "tiles/" in fp:
                    region_id = fp.split("tiles/")[1].split("/")[0]

            # Collect timeline dates
            available_dates = sorted(list({str(r.get("acquisition_date"))[:10] for r in all_sibling_records}))

            # STAGE 1: Insert into change_runs
            cur.execute("""
                INSERT INTO change_runs (
                    run_id,
                    target_tile_id,
                    site_key,
                    region_id,
                    geometry,
                    centroid_lat,
                    centroid_lon,
                    total_timeline_epochs,
                    available_dates,
                    quality_total_seen,
                    quality_passed_count,
                    quality_dropped_count,
                    staging_dir,
                    manifest_path,
                    status,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    ST_GeomFromText(%s, 4326),
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, NOW()
                );
            """, (
                run_id,
                target_tile_id,
                site_key,
                region_id,
                geom_wkt,
                c_lat,
                c_lon,
                len(all_sibling_records),
                json.dumps(available_dates),
                quality_audit.get("total_evaluated", len(all_sibling_records)),
                quality_audit.get("passed_count", 0),
                quality_audit.get("dropped_count", 0),
                staging_dir,
                manifest_path,
                "completed"
            ))

            # STAGE 2: Insert into change_quality_audits (STAY and DROPPED)
            selected_items = quality_audit.get("selected_tiles", [])
            dropped_items = quality_audit.get("dropped_tiles", [])

            # Index records by tile_id
            sib_map = {r.get("tile_id"): r for r in all_sibling_records}

            for item in selected_items:
                tid = item.get("tile_id")
                raw = sib_map.get(tid, {})
                acq_d = raw.get("acquisition_date") or item.get("date")
                year_val = int(str(acq_d)[:4]) if acq_d else None
                cur.execute("""
                    INSERT INTO change_quality_audits (
                        run_id, tile_id, acquisition_date, year,
                        cloud_pct, bad_pixels, total_pixels, bad_pixel_pct,
                        usable_pct, decision, drop_reasons, created_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, NOW()
                    );
                """, (
                    run_id, tid, acq_d, year_val,
                    float(raw.get("cloud_pct") or 0.0),
                    item.get("bad_pixels", 0),
                    item.get("total_pixels", 262144),
                    item.get("bad_pixel_pct", 0.0),
                    item.get("usable_pct", 1.0),
                    "STAY",
                    json.dumps([])
                ))

            for item in dropped_items:
                tid = item.get("tile_id")
                raw = sib_map.get(tid, {})
                acq_d = raw.get("acquisition_date") or item.get("date")
                year_val = int(str(acq_d)[:4]) if acq_d else None
                cur.execute("""
                    INSERT INTO change_quality_audits (
                        run_id, tile_id, acquisition_date, year,
                        cloud_pct, bad_pixels, total_pixels, bad_pixel_pct,
                        usable_pct, decision, drop_reasons, created_at
                    ) VALUES (
                        %s, %s, %s, %s,
                        %s, %s, %s, %s,
                        %s, %s, %s, NOW()
                    );
                """, (
                    run_id, tid, acq_d, year_val,
                    float(raw.get("cloud_pct") or 0.0),
                    item.get("bad_pixels", 0),
                    item.get("total_pixels", 262144),
                    item.get("bad_pixel_pct", 0.0),
                    item.get("usable_pct", 0.0),
                    "DROPPED",
                    json.dumps(item.get("reasons", ["Failed quality gate threshold"]))
                ))

            # STAGE 3: Insert into change_temporal_splits
            cur.execute("""
                INSERT INTO change_temporal_splits (
                    run_id,
                    method,
                    total_clean_epochs,
                    optimal_split_index,
                    step_score,
                    split_date_before,
                    split_date_after,
                    onset_bracket,
                    onset_bracket_days,
                    device,
                    device_name,
                    all_split_scores,
                    created_at
                ) VALUES (
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s, NOW()
                );
            """, (
                run_id,
                splitting_audit.get("method", "prithvi_temporal_step"),
                splitting_audit.get("total_clean_epochs_evaluated", 0),
                splitting_audit.get("optimal_split_index", 0),
                splitting_audit.get("step_score"),
                splitting_audit.get("date_before"),
                splitting_audit.get("date_after"),
                splitting_audit.get("onset_bracket"),
                splitting_audit.get("onset_bracket_days"),
                splitting_audit.get("device", "cpu"),
                splitting_audit.get("device_name", "CPU"),
                json.dumps(splitting_audit.get("all_candidates", []))
            ))

            # STAGE 4: Insert into change_clusters
            if patch_change_analysis and "cluster_semantics" in patch_change_analysis:
                cluster_dict = patch_change_analysis.get("cluster_semantics", {})
                for cid_str, cinfo in cluster_dict.items():
                    cid = int(cid_str) if str(cid_str).isdigit() else 0
                    cur.execute("""
                        INSERT INTO change_clusters (
                            run_id,
                            cluster_id,
                            predicted_type,
                            cluster_transition,
                            confidence_pct,
                            patch_count,
                            area_m2,
                            mean_delta_ndbi,
                            mean_delta_ndvi,
                            mean_delta_ndwi,
                            mean_delta_bsi,
                            interpretation,
                            ai_model,
                            created_at
                        ) VALUES (
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, NOW()
                        );
                    """, (
                        run_id,
                        cid,
                        cinfo.get("predicted_type", "Unknown"),
                        cinfo.get("cluster_transition", "Unknown"),
                        cinfo.get("confidence_pct", 0.0),
                        cinfo.get("patch_count", 0),
                        cinfo.get("area_m2", 0.0),
                        cinfo.get("mean_delta_ndbi"),
                        cinfo.get("mean_delta_ndvi"),
                        cinfo.get("mean_delta_ndwi"),
                        cinfo.get("mean_delta_bsi"),
                        cinfo.get("interpretation"),
                        cinfo.get("ai_model", "RemoteCLIP ViT-L/14 + Multi-Spectral Band Values")
                    ))

            # STAGE 5: Insert into change_candidate_patches with PostGIS Geometries!
            if patch_change_analysis and "candidates" in patch_change_analysis:
                candidates_list = patch_change_analysis.get("candidates", [])
                before_tid = staged_epochs[0]["tile_id"] if len(staged_epochs) > 0 else None
                after_tid = staged_epochs[1]["tile_id"] if len(staged_epochs) > 1 else None

                for cand in candidates_list:
                    pid = cand.get("patch_id", 0)
                    cand_id = f"cand_{run_id}_p{pid}"
                    bbox = cand.get("bbox", [0, 0, 64, 64])

                    # Compute exact WGS84 PostGIS polygon from tile footprint
                    patch_wkt, patch_lat, patch_lon = compute_patch_polygon_wkt(
                        min_lon, min_lat, max_lon, max_lat, bbox
                    )

                    ind_t1 = cand.get("indices_t1") or {}
                    ind_t2 = cand.get("indices_t2") or {}
                    deltas = cand.get("deltas") or {}

                    # Robust confidence percentage calculation (0-100)
                    conf_pct = cand.get("confidence_pct")
                    if not conf_pct or float(conf_pct) == 0.0:
                        c_val = cand.get("confidence")
                        if c_val is not None:
                            conf_pct = round(float(c_val) * 100.0, 1) if float(c_val) <= 1.0 else round(float(c_val), 1)
                        else:
                            z = cand.get("z_score") or 2.5
                            conf_pct = round(min(98.5, max(68.0, 52.0 + abs(float(z)) * 12.0)), 1)
                    else:
                        conf_pct = round(float(conf_pct), 1)

                    cur.execute("""
                        INSERT INTO change_candidate_patches (
                            candidate_id,
                            run_id,
                            patch_id,
                            grid_row,
                            grid_col,
                            bbox_px,
                            geometry,
                            centroid_lat,
                            centroid_lon,
                            cluster_id,
                            distance,
                            z_score,
                            predicted_type,
                            transition_label,
                            confidence_pct,
                            ndvi_t1, ndvi_t2, delta_ndvi,
                            ndwi_t1, ndwi_t2, delta_ndwi,
                            ndbi_t1, ndbi_t2, delta_ndbi,
                            bad_frac_t1, bad_frac_t2,
                            source_tile_before,
                            source_tile_after,
                            created_at
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s,
                            ST_GeomFromText(%s, 4326),
                            %s, %s, %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s, %s,
                            %s, %s,
                            %s, %s, NOW()
                        ) ON CONFLICT (candidate_id) DO NOTHING;
                    """, (
                        cand_id,
                        run_id,
                        pid,
                        cand.get("row", 0),
                        cand.get("col", 0),
                        bbox,
                        patch_wkt,
                        patch_lat,
                        patch_lon,
                        cand.get("cluster_id"),
                        cand.get("distance", 0.0),
                        cand.get("z_score", 0.0),
                        cand.get("predicted_type"),
                        cand.get("transition_label"),
                        conf_pct,
                        ind_t1.get("ndvi"), ind_t2.get("ndvi"), deltas.get("delta_ndvi"),
                        ind_t1.get("ndwi"), ind_t2.get("ndwi"), deltas.get("delta_ndwi"),
                        ind_t1.get("ndbi"), ind_t2.get("ndbi"), deltas.get("delta_ndbi"),
                        cand.get("bad_frac_t1"), cand.get("bad_frac_t2"),
                        before_tid,
                        after_tid
                    ))

            conn.commit()
            logger.info(f"Successfully persisted complete change run {run_id} into PostgreSQL / PostGIS.")
            return run_id

    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to persist change pipeline run: {e}", exc_info=True)
        raise e
    finally:
        conn.close()


def get_change_runs_list(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves high-level summary of all past change analysis pipeline runs."""
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    r.run_id,
                    r.target_tile_id,
                    r.site_key,
                    r.region_id,
                    r.centroid_lat,
                    r.centroid_lon,
                    r.total_timeline_epochs,
                    r.quality_passed_count,
                    r.quality_dropped_count,
                    r.staging_dir,
                    r.manifest_path,
                    r.status,
                    r.created_at,
                    ST_AsGeoJSON(r.geometry) as geom_json,
                    s.method as split_method,
                    s.onset_bracket,
                    s.split_date_before,
                    s.split_date_after,
                    s.step_score,
                    (SELECT count(*) FROM change_clusters c WHERE c.run_id = r.run_id) as cluster_count,
                    (SELECT count(*) FROM change_candidate_patches p WHERE p.run_id = r.run_id) as candidate_count
                FROM change_runs r
                LEFT JOIN change_temporal_splits s ON r.run_id = s.run_id
                ORDER BY r.created_at DESC
                LIMIT %s;
            """, (limit,))
            return cur.fetchall()
    finally:
        conn.close()


def get_change_run_details(run_id: str) -> Optional[Dict[str, Any]]:
    """Retrieves full relational data for a specific change analysis run."""
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    r.*,
                    ST_AsGeoJSON(r.geometry) as geom_json
                FROM change_runs r
                WHERE r.run_id = %s
                LIMIT 1;
            """, (run_id,))
            run_row = cur.fetchone()
            if not run_row:
                return None

            # Quality Audits
            cur.execute("""
                SELECT * FROM change_quality_audits
                WHERE run_id = %s
                ORDER BY acquisition_date ASC;
            """, (run_id,))
            quality_rows = cur.fetchall()

            # Temporal Split
            cur.execute("""
                SELECT * FROM change_temporal_splits
                WHERE run_id = %s
                LIMIT 1;
            """, (run_id,))
            split_row = cur.fetchone()

            # Clusters
            cur.execute("""
                SELECT * FROM change_clusters
                WHERE run_id = %s
                ORDER BY cluster_id ASC;
            """, (run_id,))
            cluster_rows = cur.fetchall()

            # Candidate Patches
            cur.execute("""
                SELECT 
                    p.*,
                    ST_AsGeoJSON(p.geometry) as geom_json
                FROM change_candidate_patches p
                WHERE p.run_id = %s
                ORDER BY p.distance DESC;
            """, (run_id,))
            patch_rows = cur.fetchall()

            return {
                "run": run_row,
                "quality_audits": quality_rows,
                "temporal_split": split_row,
                "clusters": cluster_rows,
                "candidate_patches": patch_rows
            }
    finally:
        conn.close()


def export_candidates_geojson(run_id: str) -> Dict[str, Any]:
    """
    Exports candidate patches for a change run as standard GeoJSON FeatureCollection
    with full properties, conforming to the NetSight export specification.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    r.site_key,
                    r.region_id,
                    s.onset_bracket,
                    s.split_date_before,
                    s.split_date_after,
                    p.candidate_id,
                    p.patch_id,
                    p.grid_row,
                    p.grid_col,
                    p.cluster_id,
                    p.distance,
                    p.z_score,
                    p.predicted_type,
                    p.transition_label,
                    p.confidence_pct,
                    p.ndvi_t1, p.ndvi_t2, p.delta_ndvi,
                    p.ndwi_t1, p.ndwi_t2, p.delta_ndwi,
                    p.ndbi_t1, p.ndbi_t2, p.delta_ndbi,
                    p.bad_frac_t1, p.bad_frac_t2,
                    p.source_tile_before,
                    p.source_tile_after,
                    ST_AsGeoJSON(p.geometry) as geom_json
                FROM change_candidate_patches p
                JOIN change_runs r ON p.run_id = r.run_id
                LEFT JOIN change_temporal_splits s ON p.run_id = s.run_id
                WHERE p.run_id = %s
                ORDER BY p.distance DESC;
            """, (run_id,))
            rows = cur.fetchall()

            features = []
            for r in rows:
                geom = json.loads(r["geom_json"]) if r.get("geom_json") else None
                props = {k: v for k, v in r.items() if k != "geom_json"}
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props
                })

            return {
                "type": "FeatureCollection",
                "run_id": run_id,
                "total_features": len(features),
                "features": features
            }
    finally:
        conn.close()
