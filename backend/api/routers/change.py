"""
backend/api/routers/change.py
=============================
Change Detection API Router:
1. /grid-map: Sovereign multi-region offline mosaic grid
2. /stage-tile: Quality Check Gate & Staging engine (drops cloudy tiles, stages clean tiles, generates manifest.json)
3. /pair: Pixel-precise change detection using per-pixel bad-mask exclusion
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Response
import psycopg2.extras

from backend.ingestion.db_writer import get_pg_connection
from backend.services.change_pair import resolve_change_pair
from backend.services.change_stager import get_all_regions_grid_map, stage_tile_temporal_series
from backend.services.quality_auditor import quality_engine

logger = logging.getLogger("change_router")
router = APIRouter(prefix="/api/v1/change", tags=["Change Detection"])


class TilePairChangeRequest(BaseModel):
    tile_id_before: Optional[str] = Field(None, description="Database tile_id for baseline (t1)")
    tile_id_after: Optional[str] = Field(None, description="Database tile_id for subsequent (t2)")
    tile_before_path: Optional[str] = Field(None, description="Direct file path to baseline GeoTIFF")
    tile_after_path: Optional[str] = Field(None, description="Direct file path to subsequent GeoTIFF")
    mask_before_path: Optional[str] = Field(None, description="Direct file path to baseline bad-mask GeoTIFF")
    mask_after_path: Optional[str] = Field(None, description="Direct file path to subsequent bad-mask GeoTIFF")
    min_usable_fraction: float = Field(0.30, ge=0.05, le=0.95, description="Minimum clear ground fraction required")
    diff_threshold: float = Field(0.15, ge=0.01, le=1.0, description="Fixed-scale reflectance change threshold")


class StageTileRequest(BaseModel):
    tile_id: str = Field(..., description="Target tile ID to stage multi-temporal series for")
    staging_dir: str = Field("change_staging", description="Target folder name under data/")


@router.get("/grid-map", response_model=Dict[str, Any])
def get_grid_map():
    """
    Returns all ingested regions and tiles formatted for the sovereign canvas.
    """
    try:
        return get_all_regions_grid_map()
    except Exception as e:
        logger.error(f"Error assembling grid map: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grid map assembly failed: {str(e)}")


@router.post("/stage-tile", response_model=Dict[str, Any])
def stage_tile(request: StageTileRequest):
    """
    Passes all temporal observations through the Quality Check Engine (inspecting mask.tif).
    DROPS tiles exceeding the bad pixel threshold (>15%), and STAYS/copies only validated
    clean observations into data/change_staging/<site_key>/.
    Writes manifest.json detailing selected_tiles and dropped_tiles.
    """
    try:
        return stage_tile_temporal_series(
            tile_id=request.tile_id,
            staging_dir_name=request.staging_dir
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error staging tile {request.tile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Staging failed: {str(e)}")


@router.post("/pair", response_model=Dict[str, Any])
def compute_pair_change(request: TilePairChangeRequest):
    """
    Computes pixel-precise multi-temporal change between two tiles using per-pixel bad-mask exclusion.
    """
    try:
        tif_before = request.tile_before_path
        tif_after = request.tile_after_path
        mask_before = request.mask_before_path
        mask_after = request.mask_after_path

        # If tile IDs are supplied, look up paths from PostgreSQL
        if request.tile_id_before and request.tile_id_after:
            conn = get_pg_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT tile_id, file_path, bad_mask_path FROM tiles WHERE tile_id IN (%s, %s);",
                    (request.tile_id_before, request.tile_id_after)
                )
                rows = {r["tile_id"]: r for r in cur.fetchall()}
            conn.close()

            if request.tile_id_before not in rows or request.tile_id_after not in rows:
                raise HTTPException(status_code=404, detail="One or both tile IDs not found in database.")

            tif_before = rows[request.tile_id_before]["file_path"]
            mask_before = rows[request.tile_id_before].get("bad_mask_path")
            tif_after = rows[request.tile_id_after]["file_path"]
            mask_after = rows[request.tile_id_after].get("bad_mask_path")

        if not tif_before or not tif_after:
            raise HTTPException(
                status_code=400,
                detail="Must provide either (tile_id_before and tile_id_after) or (tile_before_path and tile_after_path)."
            )

        report = resolve_change_pair(
            tile_before_path=tif_before,
            tile_after_path=tif_after,
            mask_before_path=mask_before,
            mask_after_path=mask_after,
            min_usable_fraction=request.min_usable_fraction,
            diff_threshold=request.diff_threshold
        )

        return {
            "status": "success",
            "tile_before": request.tile_id_before or str(tif_before),
            "tile_after": request.tile_id_after or str(tif_after),
            "result": report
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change pair evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Change detection failed: {str(e)}")


@router.get("/runs", response_model=Dict[str, Any])
def list_change_runs(limit: int = 50):
    """
    Lists all change analysis pipeline runs stored in PostgreSQL / PostGIS.
    """
    try:
        from backend.services.change_persistence import get_change_runs_list
        runs = get_change_runs_list(limit=limit)
        return {
            "status": "success",
            "total_runs": len(runs),
            "runs": runs
        }
    except Exception as e:
        logger.error(f"Error retrieving change runs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch change runs: {str(e)}")


@router.get("/runs/{run_id}", response_model=Dict[str, Any])
def get_change_run(run_id: str):
    """
    Retrieves full stage-by-stage relational history for a specific change pipeline run:
    - Target tile & timeline
    - Quality audit (STAY vs DROPPED per epoch)
    - Temporal splitting / onset bracket
    - Change clusters & semantic typing
    - Candidate patches with PostGIS geometry
    """
    try:
        from backend.services.change_persistence import get_change_run_details
        details = get_change_run_details(run_id)
        if not details:
            raise HTTPException(status_code=404, detail=f"Change run {run_id} not found.")
        return {
            "status": "success",
            "run_id": run_id,
            "data": details
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving change run details for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch run details: {str(e)}")


@router.get("/runs/{run_id}/candidates.geojson")
def export_run_candidates_geojson(run_id: str):
    """
    Exports all candidate patches for a change run as standard GeoJSON FeatureCollection,
    including real-world geospatial polygons, cluster assignments, and spectral deltas.
    """
    try:
        from backend.services.change_persistence import export_candidates_geojson
        return export_candidates_geojson(run_id)
    except Exception as e:
        logger.error(f"Error exporting candidates GeoJSON for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to export GeoJSON: {str(e)}")


# =============================================================================
# Analyst Review Workflow Endpoints (Change Detection)
# =============================================================================

class AnalystChangeDecisionRequest(BaseModel):
    candidate_id: str
    decision: str = Field(..., description="'confirmed', 'rejected', or 'unsure'")
    analyst_id: str = Field("ANALYST-DEF-01", description="Identifier of the reviewing analyst")
    note: str = Field("", description="Intelligence note or justification")
    tags: Optional[list] = Field(default_factory=list, description="Optional classification tags")


@router.get("/runs-summary", response_model=Dict[str, Any])
def get_runs_review_summary():
    """
    Returns high-level summary of all change runs and analyst review progress.
    """
    try:
        from backend.services.analyst_workflow import get_change_runs_summary
        runs = get_change_runs_summary()
        return {"status": "success", "runs": runs}
    except Exception as e:
        logger.error(f"Error fetching runs summary: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/review-queue", response_model=Dict[str, Any])
def get_run_review_queue(run_id: str):
    """
    Returns candidate patches for run_id with priority ranking and current review status.
    """
    try:
        from backend.services.analyst_workflow import get_change_review_queue
        queue = get_change_review_queue(run_id)
        return {
            "status": "success",
            "run_id": run_id,
            "total_candidates": len(queue),
            "queue": queue
        }
    except Exception as e:
        logger.error(f"Error fetching review queue for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/runs/{run_id}/decide", response_model=Dict[str, Any])
def submit_change_decision(run_id: str, request: AnalystChangeDecisionRequest):
    """
    Appends an immutable analyst decision record (confirmed, rejected, unsure).
    """
    try:
        from backend.services.analyst_workflow import record_change_decision
        decision = record_change_decision(
            run_id=run_id,
            candidate_id=request.candidate_id,
            decision=request.decision,
            analyst_id=request.analyst_id,
            note=request.note,
            tags=request.tags
        )
        return {"status": "success", "decision": decision}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Error recording decision for {run_id}/{request.candidate_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/candidates/{candidate_id}/history", response_model=Dict[str, Any])
def get_candidate_decision_history(run_id: str, candidate_id: str):
    """
    Returns complete immutable audit history for a specific candidate patch.
    """
    try:
        from backend.services.analyst_workflow import get_change_candidate_history
        history = get_change_candidate_history(run_id, candidate_id)
        return {"status": "success", "candidate_id": candidate_id, "history": history}
    except Exception as e:
        logger.error(f"Error fetching candidate history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/export.geojson")
def export_change_review_geojson(run_id: str, status: Optional[str] = None):
    """
    Exports candidates with latest analyst decisions and full cryptographic provenance.
    Filter by status: e.g. status=confirmed or omit for all.
    """
    try:
        from backend.services.analyst_workflow import export_change_geojson
        statuses = [status] if status else None
        return export_change_geojson(run_id, statuses)
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error exporting GeoJSON for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs/{run_id}/export.pdf")
def export_change_review_pdf(run_id: str):
    """
    Exports a high-fidelity Defense Intelligence PDF Dossier for run_id.
    Includes embedded comparative satellite imagery, change footprints,
    cluster dynamics, and analyst audit records.
    """
    try:
        from backend.services.pdf_report_generator import generate_change_dossier_pdf
        pdf_bytes = generate_change_dossier_pdf(run_id)
        filename = f"Change_Intelligence_Dossier_{run_id}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"'
            }
        )
    except Exception as e:
        logger.error(f"Error generating PDF dossier for {run_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))





@router.get("/runs/{run_id}/export.geotiff")
def export_change_review_geotiff(run_id: str):
    """
    Exports all georeferenced GeoTIFF assets associated with run_id (baseline, milestone epochs,
    and change candidate masks) as a compressed archive (.zip) for GIS workflows.
    """
    import io
    import zipfile
    from pathlib import Path

    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT staging_dir, manifest_path FROM change_runs WHERE run_id = %s;", (run_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Run {run_id} not found.")

            # Resolve staging path safely
            raw_stage = str(row["staging_dir"]).replace("\\", "/").strip("/")
            repo_root = Path(__file__).resolve().parents[3]
            staging_path = repo_root / raw_stage
            if not staging_path.is_dir():
                staging_path = Path("/app") / raw_stage
            if not staging_path.is_dir():
                raise HTTPException(status_code=404, detail=f"Staging directory for {run_id} not found on disk ({staging_path}).")

            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in staging_path.rglob("*"):
                    if file_path.is_file():
                        rel_path = file_path.relative_to(staging_path)
                        zip_file.write(file_path, arcname=str(rel_path))

            buf.seek(0)
            filename = f"Change_GeoTIFF_Bundle_{run_id}.zip"
            return Response(
                content=buf.getvalue(),
                media_type="application/zip",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}"'
                }
            )
    finally:
        conn.close()


