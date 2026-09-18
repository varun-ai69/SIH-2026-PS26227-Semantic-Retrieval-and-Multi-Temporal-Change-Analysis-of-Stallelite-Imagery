"""
backend/services/change_stager.py
=================================
Multi-Temporal Tile Staging & Quality Gate Engine
PS Sections: 2.2.1 / 2.2.2 / 2.2.3

Workflow:
  1. get_all_regions_grid_map:
     Queries PostGIS/PostgreSQL tiles and ingestion_coverage to assemble the
     multi-region offline mosaic canvas for frontend change.html.

  2. stage_tile_temporal_series:
     - Given a target tile, retrieves all historical/contemporary acquisitions for that ground footprint (`site_key`).
     - Passes ALL tiles through the Quality Check Engine (`QualityCheckEngine`) which inspects `_mask.tif`.
     - DROPS / EXCLUDES tiles exceeding the bad-pixel / cloud threshold (e.g. >15% bad pixels).
     - ONLY STAYS / COPIES the validated clean tiles into:
       `data/<staging_dir>/<site_key>/T<idx>_<date>/`
     - Outputs `manifest.json` with a full audit log detailing:
       * `selected_tiles` (STAY) with bad pixel count & usability %
       * `dropped_tiles` (DROPPED) with bad pixel count & rejection reasons
"""

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

import psycopg2.extras
from backend.ingestion.db_writer import get_pg_connection
from backend.services.quality_auditor import quality_engine
from backend.services.temporal_splitter import get_temporal_splitter
from backend.services.patch_change_detector import get_patch_change_detector

logger = logging.getLogger("change_stager")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_file_path(path_str: Optional[str]) -> Optional[Path]:
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p
    p_norm = str(path_str).replace("\\", "/")
    idx = p_norm.find("data/")
    if idx != -1:
        cand = REPO_ROOT / p_norm[idx:]
        if cand.is_file():
            return cand
    return None


def _to_web_url(file_path: Optional[str]) -> Optional[str]:
    """Converts a local disk path into a web-accessible static URL (/data/...)."""
    if not file_path:
        return None
    p = str(file_path).replace("\\", "/")
    idx = p.find("data/")
    if idx != -1:
        return "/" + p[idx:]
    if p.startswith("/app/data/"):
        return p.replace("/app/data/", "/data/")
    return "/" + p.lstrip("/")


def get_all_regions_grid_map() -> Dict[str, Any]:
    """
    Queries PostgreSQL to retrieve all tiles across all ingested regions,
    grouped by region and spatial site_key.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Fetch coverage regions
            cur.execute("""
                SELECT 
                    region_id,
                    region_name,
                    status,
                    tile_count,
                    last_updated,
                    ST_AsGeoJSON(geometry) as geom_json,
                    ST_XMin(geometry) as min_lon,
                    ST_YMin(geometry) as min_lat,
                    ST_XMax(geometry) as max_lon,
                    ST_YMax(geometry) as max_lat
                FROM ingestion_coverage
                ORDER BY last_updated DESC;
            """)
            coverage_rows = cur.fetchall()

            # 2. Fetch all Sentinel-2 tiles with bounding coordinates (exclude Maxar high-res)
            cur.execute("""
                SELECT 
                    t.tile_id,
                    t.site_key,
                    t.scene_id,
                    t.sensor,
                    t.acquisition_date,
                    t.centroid_lat,
                    t.centroid_lon,
                    t.cloud_pct,
                    t.quality_confidence,
                    t.mean_ndvi,
                    t.mean_ndwi,
                    t.mean_ndbi,
                    t.file_path,
                    t.bad_mask_path,
                    t.thumbnail_path,
                    ST_AsGeoJSON(t.geometry) as geom_json,
                    ST_XMin(t.geometry) as min_lon,
                    ST_YMin(t.geometry) as min_lat,
                    ST_XMax(t.geometry) as max_lon,
                    ST_YMax(t.geometry) as max_lat
                FROM tiles t
                WHERE (t.sensor ILIKE 'Sentinel%%' OR t.sensor IS NULL)
                  AND (t.file_path NOT ILIKE '%%maxar%%' OR t.file_path IS NULL)
                ORDER BY t.acquisition_date DESC, t.tile_id ASC;
            """)
            tile_rows = cur.fetchall()

            # 3. Compute temporal depth per site_key
            site_epoch_counts: Dict[str, int] = {}
            site_dates: Dict[str, List[str]] = {}
            for tr in tile_rows:
                sk = tr.get("site_key") or tr["tile_id"]
                d_str = str(tr["acquisition_date"])[:10]
                if sk not in site_dates:
                    site_dates[sk] = []
                if d_str not in site_dates[sk]:
                    site_dates[sk].append(d_str)
                site_epoch_counts[sk] = len(site_dates[sk])

    finally:
        conn.close()

    # Organize regions
    regions_map: Dict[str, Dict[str, Any]] = {}
    for cr in coverage_rows:
        rid = cr["region_id"]
        regions_map[rid] = {
            "region_id": rid,
            "region_name": cr["region_name"] or rid,
            "bbox": [float(cr["min_lon"]), float(cr["min_lat"]), float(cr["max_lon"]), float(cr["max_lat"])],
            "center": [(float(cr["min_lat"]) + float(cr["max_lat"])) / 2.0, (float(cr["min_lon"]) + float(cr["max_lon"])) / 2.0],
            "tile_count": cr["tile_count"],
            "dates": []
        }

    formatted_tiles: List[Dict[str, Any]] = []
    region_date_sets: Dict[str, set] = {rid: set() for rid in regions_map}

    for tr in tile_rows:
        sk = tr.get("site_key") or tr["tile_id"]
        lat = float(tr["centroid_lat"])
        lon = float(tr["centroid_lon"])
        acq_date = str(tr["acquisition_date"])
        date_short = acq_date[:10]

        # Determine region
        assigned_region_id = "global"
        for rid, rdata in regions_map.items():
            b = rdata["bbox"]
            if b[0] - 0.05 <= lon <= b[2] + 0.05 and b[1] - 0.05 <= lat <= b[3] + 0.05:
                assigned_region_id = rid
                region_date_sets[rid].add(date_short)
                break

        bounds_leaflet = [
            [float(tr["min_lat"]), float(tr["min_lon"])],
            [float(tr["max_lat"]), float(tr["max_lon"])]
        ]

        formatted_tiles.append({
            "tile_id": tr["tile_id"],
            "site_key": sk,
            "scene_id": tr["scene_id"],
            "sensor": tr["sensor"] or "Sentinel-2",
            "region_id": assigned_region_id,
            "acquisition_date": acq_date,
            "date_short": date_short,
            "centroid_lat": lat,
            "centroid_lon": lon,
            "bounds": bounds_leaflet,
            "bbox_wgs84": [float(tr["min_lon"]), float(tr["min_lat"]), float(tr["max_lon"]), float(tr["max_lat"])],
            "cloud_pct": round(float(tr.get("cloud_pct") or 0.0), 2),
            "quality_confidence": round(float(tr.get("quality_confidence") or 1.0), 2),
            "mean_ndvi": round(float(tr["mean_ndvi"]), 3) if tr.get("mean_ndvi") is not None else None,
            "mean_ndwi": round(float(tr["mean_ndwi"]), 3) if tr.get("mean_ndwi") is not None else None,
            "mean_ndbi": round(float(tr["mean_ndbi"]), 3) if tr.get("mean_ndbi") is not None else None,
            "file_path": tr["file_path"],
            "bad_mask_path": tr.get("bad_mask_path"),
            "thumbnail_url": _to_web_url(tr.get("thumbnail_path")),
            "epoch_count": site_epoch_counts.get(sk, 1),
            "available_dates": sorted(site_dates.get(sk, [date_short]))
        })

    for rid, rdata in regions_map.items():
        rdata["dates"] = sorted(list(region_date_sets.get(rid, set())))

    return {
        "status": "success",
        "total_tiles": len(formatted_tiles),
        "total_regions": len(regions_map),
        "regions": list(regions_map.values()),
        "tiles": formatted_tiles
    }


def stage_tile_temporal_series(
    tile_id: str,
    staging_dir_name: str = "change_staging"
) -> Dict[str, Any]:
    """
    Stages all temporal epochs for a given tile's footprint after running
    the Quality Check Engine.

    Only STAY tiles (passing quality audit) are copied to disk.
    DROPPED tiles are recorded in `manifest.json` under `quality_audit.dropped_tiles`.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Get the target tile's site_key, sensor, and file_path
            cur.execute("SELECT site_key, sensor, file_path, centroid_lat, centroid_lon FROM tiles WHERE tile_id = %s LIMIT 1;", (tile_id,))
            target = cur.fetchone()
            if not target:
                raise ValueError(f"Tile {tile_id} not found in database.")

            # Strict guard: Exclude Maxar high-res tiles from Sentinel multi-spectral change detection
            t_sensor = str(target.get("sensor") or "").lower()
            t_fp = str(target.get("file_path") or "").lower()
            if "maxar" in t_sensor or "maxar" in t_fp or "worldview" in t_sensor:
                raise ValueError(
                    f"Tile {tile_id} is a sub-meter optical Maxar tile ({target.get('sensor')}). "
                    f"Change detection pipeline strictly requires Sentinel-2 10m multi-spectral observations."
                )

            site_key = target.get("site_key") or tile_id

            # 2. Fetch all temporal sibling tiles sharing this site_key (strictly Sentinel-2)
            cur.execute("""
                SELECT 
                    tile_id,
                    site_key,
                    scene_id,
                    sensor,
                    acquisition_date,
                    centroid_lat,
                    centroid_lon,
                    cloud_pct,
                    quality_confidence,
                    mean_ndvi,
                    mean_ndwi,
                    mean_ndbi,
                    file_path,
                    bad_mask_path,
                    thumbnail_path
                FROM tiles
                WHERE site_key = %s
                  AND (sensor ILIKE 'Sentinel%%' OR sensor IS NULL)
                  AND (file_path NOT ILIKE '%%maxar%%' OR file_path IS NULL)
                ORDER BY acquisition_date ASC;
            """, (site_key,))
            sibling_rows = cur.fetchall()

    finally:
        conn.close()

    if not sibling_rows:
        raise ValueError(f"No temporal observations found for site_key {site_key}")

    # ============================================================
    # 3. RUN QUALITY CHECK ENGINE ON ALL CANDIDATE TILES
    # ============================================================
    audit_report = quality_engine.audit_temporal_series(site_key, sibling_rows)
    selected_tiles = audit_report["selected_tiles"]
    dropped_tiles = audit_report["dropped_tiles"]

    # If all tiles were dropped (e.g. extremely cloudy), fallback to the best available 2
    if len(selected_tiles) == 0 and len(dropped_tiles) > 0:
        logger.warning(f"All observations dropped by quality gate for {site_key}. Selecting best available as fallback.")
        fallback_sorted = sorted(dropped_tiles, key=lambda d: d["usable_pct"], reverse=True)
        selected_tiles = fallback_sorted[:2]
        dropped_ids = {s["tile_id"] for s in selected_tiles}
        dropped_tiles = [d for d in dropped_tiles if d["tile_id"] not in dropped_ids]

    if len(selected_tiles) < 2:
        raise ValueError(f"Insufficient usable observations for site_key {site_key} (need at least 2 clean tiles).")

    # ============================================================
    # 4. RUN PRITHVI TEMPORAL SPLITTING ENGINE
    # ============================================================
    clean_records = [s["raw_record"] for s in selected_tiles]
    splitter = get_temporal_splitter()
    split_res = splitter.evaluate_series(clean_records)

    rec_before = split_res["selected_before_record"]
    rec_after = split_res["selected_after_record"]

    # Quick lookup for quality stats of selected milestone records
    sel_lookup = {s["tile_id"]: s for s in selected_tiles}
    sel_before = sel_lookup.get(rec_before["tile_id"], selected_tiles[0])
    sel_after = sel_lookup.get(rec_after["tile_id"], selected_tiles[-1])

    # Target staging directory: data/change_staging/<site_key>/
    base_staging_dir = REPO_ROOT / "data" / staging_dir_name / site_key
    # Strictly remove any previous runs/stale epochs to ensure only the 2 milestone folders exist
    if base_staging_dir.exists():
        shutil.rmtree(base_staging_dir, ignore_errors=True)
    base_staging_dir.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # 5. STAGE STRICTLY AND ONLY THE 2 MILESTONE TILES (T1, T2)
    # ============================================================
    staged_epochs: List[Dict[str, Any]] = []
    milestone_targets = [
        ("T1", "baseline_before", sel_before, rec_before),
        ("T2", "onset_after", sel_after, rec_after)
    ]

    for tag, role, sel_info, row in milestone_targets:
        acq_date_str = str(row["acquisition_date"])[:10]
        epoch_folder_name = f"{tag}_{acq_date_str}"
        epoch_dir = base_staging_dir / epoch_folder_name
        epoch_dir.mkdir(parents=True, exist_ok=True)

        src_tif = resolve_file_path(row.get("file_path"))
        src_mask = resolve_file_path(row.get("bad_mask_path"))
        src_thumb = resolve_file_path(row.get("thumbnail_path"))

        dst_tif = epoch_dir / "tile.tif"
        dst_mask = epoch_dir / "mask.tif"
        dst_thumb = epoch_dir / "thumb.jpg"

        if src_tif and src_tif.is_file():
            shutil.copy2(src_tif, dst_tif)
        if src_mask and src_mask.is_file():
            shutil.copy2(src_mask, dst_mask)
        if src_thumb and src_thumb.is_file():
            shutil.copy2(src_thumb, dst_thumb)

        staged_epoch = {
            "epoch": tag,
            "role": role,
            "tile_id": row["tile_id"],
            "scene_id": row["scene_id"],
            "acquisition_date": str(row["acquisition_date"]),
            "date_short": acq_date_str,
            "sensor": row.get("sensor") or "Sentinel-2",
            "cloud_pct": float(row.get("cloud_pct") or 0.0),
            "quality_confidence": float(row.get("quality_confidence") or 1.0),
            "mean_ndvi": float(row["mean_ndvi"]) if row.get("mean_ndvi") is not None else None,
            "mean_ndwi": float(row["mean_ndwi"]) if row.get("mean_ndwi") is not None else None,
            "mean_ndbi": float(row["mean_ndbi"]) if row.get("mean_ndbi") is not None else None,
            "quality_check": {
                "bad_pixels": sel_info["bad_pixels"],
                "total_pixels": sel_info["total_pixels"],
                "bad_pixel_pct": sel_info["bad_pixel_pct"],
                "usable_pct": sel_info["usable_pct"],
                "status": "STAY"
            },
            "files": {
                "folder": str(epoch_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
                "tif": str(dst_tif.relative_to(REPO_ROOT)).replace("\\", "/"),
                "mask": str(dst_mask.relative_to(REPO_ROOT)).replace("\\", "/"),
                "thumb": str(dst_thumb.relative_to(REPO_ROOT)).replace("\\", "/"),
                "thumb_url": _to_web_url(str(dst_thumb))
            }
        }
        staged_epochs.append(staged_epoch)

    # 6. Compute spectral deltas between milestone T1 and T2
    deltas = {}
    if len(staged_epochs) == 2:
        t_pre = staged_epochs[0]
        t_post = staged_epochs[1]

        if t_pre["mean_ndvi"] is not None and t_post["mean_ndvi"] is not None:
            deltas["delta_ndvi"] = round(t_post["mean_ndvi"] - t_pre["mean_ndvi"], 4)
        if t_pre["mean_ndwi"] is not None and t_post["mean_ndwi"] is not None:
            deltas["delta_ndwi"] = round(t_post["mean_ndwi"] - t_pre["mean_ndwi"], 4)
        if t_pre["mean_ndbi"] is not None and t_post["mean_ndbi"] is not None:
            deltas["delta_ndbi"] = round(t_post["mean_ndbi"] - t_pre["mean_ndbi"], 4)

    # 7. Assemble Quality Audit Summary
    manifest_passed = [
        {
            "tile_id": s["tile_id"],
            "date": s["date"],
            "bad_pixels": s["bad_pixels"],
            "total_pixels": s["total_pixels"],
            "bad_pixel_pct": s["bad_pixel_pct"],
            "usable_pct": s["usable_pct"],
            "status": "STAY"
        }
        for s in selected_tiles
    ]
    manifest_dropped = [
        {
            "tile_id": d["tile_id"],
            "date": d["date"],
            "bad_pixels": d["bad_pixels"],
            "total_pixels": d["total_pixels"],
            "bad_pixel_pct": d["bad_pixel_pct"],
            "usable_pct": d["usable_pct"],
            "status": "DROPPED",
            "reasons": d.get("reasons", ["Failed quality threshold"])
        }
        for d in dropped_tiles
    ]

    quality_audit_summary = {
        "thresholds_applied": audit_report["thresholds_applied"],
        "total_evaluated": audit_report["total_evaluated"],
        "passed_count": len(manifest_passed),
        "dropped_count": len(manifest_dropped),
        "selected_tiles": manifest_passed,
        "dropped_tiles": manifest_dropped
    }

    # 8. Assemble Splitting Engine Audit
    splitting_audit = {
        "method": split_res["method"],
        "device": split_res.get("device", "cpu"),
        "device_name": split_res.get("device_name", "CPU"),
        "total_clean_epochs_evaluated": split_res["total_clean_epochs"],
        "optimal_split_index": split_res["split_index"],
        "step_score": split_res["step_score"],
        "onset_bracket": split_res["onset_bracket"],
        "date_before": split_res["split_date_before"],
        "date_after": split_res["split_date_after"],
        "all_candidates": split_res.get("all_split_scores", [])
    }

    # 9. CONTINUOUS PIPELINE: Execute Prithvi Patch Change Detection on the 2 Milestone Tiles
    patch_change_report = None
    if len(staged_epochs) == 2:
        try:
            logger.info(f"Executing Prithvi Patch Change Detection on milestone tiles for site {site_key}...")
            detector = get_patch_change_detector()
            t1_epoch = staged_epochs[0]
            t2_epoch = staged_epochs[1]

            t1_tif = base_staging_dir / f"{t1_epoch['epoch']}_{t1_epoch['date_short']}" / "tile.tif"
            t2_tif = base_staging_dir / f"{t2_epoch['epoch']}_{t2_epoch['date_short']}" / "tile.tif"
            t1_mask = base_staging_dir / f"{t1_epoch['epoch']}_{t1_epoch['date_short']}" / "mask.tif"
            t2_mask = base_staging_dir / f"{t2_epoch['epoch']}_{t2_epoch['date_short']}" / "mask.tif"

            patch_change_report = detector.run_detection(
                before_tif=str(t1_tif),
                after_tif=str(t2_tif),
                before_mask=str(t1_mask) if t1_mask.is_file() else None,
                after_mask=str(t2_mask) if t2_mask.is_file() else None,
                output_dir=str(base_staging_dir),
                grid_size=8,
                quality_thresh=0.20,
                z_threshold=2.0,
                min_dist=0.01,
                min_cluster_size=2
            )
            logger.info(f"Patch change detection complete: {patch_change_report.get('num_candidates', 0)} candidate patches confirmed.")
        except Exception as pe:
            logger.error(f"Error running patch change detection on {site_key}: {pe}", exc_info=True)
            patch_change_report = {
                "status": "failed",
                "error": str(pe),
                "num_candidates": 0,
                "candidates": []
            }

    # 10. Write manifest.json
    manifest_data = {
        "manifest_version": "2.2.0",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "site_key": site_key,
        "centroid": [float(target["centroid_lat"]), float(target["centroid_lon"])],
        "staging_dir": str(base_staging_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "quality_audit": quality_audit_summary,
        "splitting_audit": splitting_audit,
        "patch_change_analysis": patch_change_report,
        "total_staged_epochs": len(staged_epochs),
        "epochs": staged_epochs,
        "spectral_deltas": deltas,
        "downstream_ready": len(staged_epochs) == 2
    }

    manifest_path = base_staging_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    # 11. Transactionally persist complete pipeline execution into PostgreSQL / PostGIS
    run_id = None
    try:
        from backend.services.change_persistence import save_change_pipeline_run
        rel_manifest = str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/")
        rel_staging = str(base_staging_dir.relative_to(REPO_ROOT)).replace("\\", "/")
        run_id = save_change_pipeline_run(
            target_tile_id=tile_id,
            site_key=site_key,
            staging_dir=rel_staging,
            manifest_path=rel_manifest,
            quality_audit=quality_audit_summary,
            splitting_audit=splitting_audit,
            patch_change_analysis=patch_change_report,
            all_sibling_records=sibling_rows,
            staged_epochs=staged_epochs,
            spectral_deltas=deltas
        )
        logger.info(f"Pipeline run {run_id} persisted in PostgreSQL/PostGIS.")
    except Exception as db_err:
        logger.error(f"Error persisting change pipeline run to database: {db_err}", exc_info=True)

    return {
        "status": "success",
        "run_id": run_id,
        "message": f"Optimal change transition identified ({split_res['onset_bracket']}) and patch change detection executed.",
        "site_key": site_key,
        "staging_dir": str(base_staging_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        "quality_audit": quality_audit_summary,
        "splitting_audit": splitting_audit,
        "patch_change_analysis": patch_change_report,
        "epochs": staged_epochs,
        "spectral_deltas": deltas,
        "manifest_path": str(manifest_path.relative_to(REPO_ROOT)).replace("\\", "/")
    }
