"""
backend/api/routers/ingest.py
=============================
Ingestion API Router — provides endpoints for:
  1. Entry Point A: AOI Polygon with timeline (e.g. 1-10 years or date ranges)
  2. Entry Point B: Offline GeoTIFF / Evaluation File Ingestion
  3. Maxar High-Res: Async background job with polling (no browser timeout)
"""

import os
import shutil
import logging
import uuid
import threading
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, UploadFile, File, Form

from backend.ingestion.pipeline import (
    run_aoi_ingestion_pipeline,
    run_direct_file_ingestion_pipeline,
    PipelineResult
)
from backend.maxar_ingestion.pipeline import run_maxar_ingestion_pipeline
from backend.services.system_mode import is_offline_mode

logger = logging.getLogger("ingest_router")
router = APIRouter(prefix="/api/v1/ingest", tags=["Ingestion"])


# ============================================================
# In-Memory Job Store for async Maxar ingestion
# job_id -> {status, progress, message, result, region_id, created_at}
# ============================================================
_JOBS: Dict[str, Dict[str, Any]] = {}
_JOBS_LOCK = threading.Lock()


def _run_maxar_job(job_id: str, derived_region: str, request_dict: Dict[str, Any]):
    """Background thread: runs Maxar pipeline and updates job store."""
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["message"] = "Connecting to Maxar Wayback Archive & selecting epochs..."
        _JOBS[job_id]["progress"] = 10

    try:
        with _JOBS_LOCK:
            _JOBS[job_id]["message"] = "Streaming WMTS tiles & stitching mosaic canvas..."
            _JOBS[job_id]["progress"] = 30

        result = run_maxar_ingestion_pipeline(
            geojson_polygon=request_dict["geojson_polygon"],
            region_id=derived_region,
            region_name=request_dict.get("region_name"),
            years=request_dict.get("years") or [2020, 2026],
            zoom_level=request_dict.get("zoom_level", 16),
            overlap_pct=request_dict.get("overlap_pct", 0.10),
            populate_db=request_dict.get("populate_db", True)
        )

        with _JOBS_LOCK:
            if result.status == "failed":
                _JOBS[job_id]["status"] = "failed"
                _JOBS[job_id]["message"] = "; ".join(result.errors) if result.errors else "Pipeline failed"
                _JOBS[job_id]["progress"] = 0
            else:
                _JOBS[job_id]["status"] = "done"
                _JOBS[job_id]["progress"] = 100
                _JOBS[job_id]["message"] = (
                    f"Done! {result.total_tiles_generated} tiles across "
                    f"{len(result.epochs_processed)} epochs in {result.execution_time_seconds:.1f}s"
                )
                _JOBS[job_id]["result"] = {
                    "status": "success",
                    "region_id": result.region_id,
                    "sensor": "Maxar WorldView / Wayback",
                    "epochs_processed": result.epochs_processed,
                    "total_tiles_generated": result.total_tiles_generated,
                    "tiles_upserted_postgres": result.tiles_upserted_postgres,
                    "vectors_upserted_qdrant": result.vectors_upserted_qdrant,
                    "qdrant_collection": "maxar_tile_embeddings",
                    "elapsed_seconds": result.execution_time_seconds,
                    "errors": result.errors
                }

    except Exception as e:
        logger.error(f"Maxar background job {job_id} failed: {e}", exc_info=True)
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["message"] = f"Pipeline error: {str(e)}"
            _JOBS[job_id]["progress"] = 0


# ============================================================
# Pydantic Request Models
# ============================================================

class AOIIngestRequest(BaseModel):
    geojson_polygon: Dict[str, Any] = Field(..., description="GeoJSON Polygon geometry or Feature")
    sensor: Optional[str] = Field("sentinel2", description="Sensor source: 'sentinel2' or 'maxar'")
    region_id: Optional[str] = Field(None, description="Unique region identifier")
    region_name: Optional[str] = Field(None, description="Human-readable region name")
    date_from: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    date_to: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    years_timeline: Optional[int] = Field(None, description="Years of historical timeline to query (e.g. 1 to 10)")
    num_time_buckets: int = Field(2, ge=1, le=10, description="Number of temporal buckets across the range")
    max_cloud_cover: float = Field(20.0, ge=0.0, le=100.0, description="Maximum cloud cover percentage")
    ground_crop_size: int = Field(512, description="Ground crop size in pixels")
    overlap_pct: float = Field(0.10, ge=0.0, le=0.5, description="Overlap percentage between tiles")
    populate_db: bool = Field(True, description="Whether to upsert into Postgres and Qdrant")


class MaxarIngestRequest(BaseModel):
    geojson_polygon: Dict[str, Any] = Field(..., description="GeoJSON Polygon geometry or Feature")
    region_id: Optional[str] = Field(None, description="Unique region identifier")
    region_name: Optional[str] = Field(None, description="Human-readable region name")
    years: Optional[List[int]] = Field(default=[2020, 2026], description="Wayback historical epochs (e.g. [2020, 2026])")
    zoom_level: int = Field(16, ge=14, le=18, description="WMTS zoom level (16=~2.4m, 17=~1.2m, 18=sub-meter)")
    overlap_pct: float = Field(0.10, ge=0.0, le=0.5, description="Overlap percentage between tiles")
    populate_db: bool = Field(True, description="Whether to upsert into Postgres and Qdrant")


class FileIngestRequest(BaseModel):
    file_path: Optional[str] = Field(None, description="Absolute or relative path to local GeoTIFF file or folder")
    file_paths: Optional[List[str]] = Field(None, description="List of paths for separate band GeoTIFFs")
    sensor: Optional[str] = Field("auto", description="Sensor convention: auto, landsat, sentinel2, or generic")
    region_id: Optional[str] = Field(None, description="Region ID")
    custom_band_order: Optional[List[str]] = Field(None, description="Custom band order e.g. ['blue', 'green', 'red', 'nir', 'swir']")
    acquisition_date: Optional[str] = Field(None, description="Acquisition date (YYYY-MM-DD)")
    ground_crop_size: int = Field(512, description="Ground crop size in pixels")
    overlap_pct: float = Field(0.10, ge=0.0, le=0.5, description="Overlap percentage between tiles")
    populate_db: bool = Field(True, description="Whether to upsert into Postgres and Qdrant")


# ============================================================
# Routes
# ============================================================

@router.post("/aoi", response_model=Dict[str, Any])
def ingest_aoi(request: AOIIngestRequest):
    """
    Entry Point A: Ingests an AOI polygon across the requested timeline.
    If sensor='maxar', launches async background job and returns job_id immediately.
    """
    if is_offline_mode():
        raise HTTPException(
            status_code=403,
            detail="Canopus is currently operating in Air-Gapped Offline Mode. External STAC / Sentinel-2 satellite discovery is disabled. Switch to Online Mode in the system settings to fetch remote imagery, or use local GeoTIFF file ingestion."
        )

    try:
        today = date.today()
        if request.years_timeline and request.years_timeline > 0:
            start_year = today.year - request.years_timeline
            date_from = f"{start_year}-01-01"
            date_to = today.isoformat()
            num_buckets = max(request.num_time_buckets, min(request.years_timeline, 8))
        else:
            date_from = request.date_from or "2023-01-01"
            date_to = request.date_to or today.isoformat()
            num_buckets = request.num_time_buckets

        region_id = request.region_id or f"region_aoi_{uuid.uuid4().hex[:8]}"

        # Route Maxar to async background job
        if request.sensor and "maxar" in request.sensor.lower():
            start_yr = int(date_from[:4]) if date_from else 2020
            end_yr = int(date_to[:4]) if date_to else 2026
            years_list = [start_yr, end_yr] if start_yr != end_yr else [start_yr]
            if len(years_list) == 1 and start_yr == 2026:
                years_list = [2020, 2026]

            job_id = uuid.uuid4().hex[:12]
            with _JOBS_LOCK:
                _JOBS[job_id] = {
                    "status": "queued",
                    "progress": 0,
                    "message": "Job queued — pipeline starting...",
                    "result": None,
                    "region_id": region_id,
                    "created_at": datetime.utcnow().isoformat()
                }

            t = threading.Thread(
                target=_run_maxar_job,
                args=(job_id, region_id, {
                    "geojson_polygon": request.geojson_polygon,
                    "region_name": request.region_name,
                    "years": years_list,
                    "zoom_level": 16,
                    "overlap_pct": request.overlap_pct,
                    "populate_db": request.populate_db
                }),
                daemon=True
            )
            t.start()
            logger.info(f"Maxar async job {job_id} started for region '{region_id}' epochs {years_list}")
            return {
                "status": "queued",
                "job_id": job_id,
                "region_id": region_id,
                "message": "Maxar ingestion started in background. Poll /api/v1/ingest/job/{job_id} for status."
            }

        logger.info(f"Starting Sentinel-2 AOI ingestion for region '{region_id}' ({date_from} to {date_to})")

        result: PipelineResult = run_aoi_ingestion_pipeline(
            geojson_input=request.geojson_polygon,
            date_from=date_from,
            date_to=date_to,
            region_id=region_id,
            region_name=request.region_name,
            num_time_buckets=num_buckets,
            max_cloud_cover=request.max_cloud_cover,
            ground_crop_size=request.ground_crop_size,
            overlap_pct=request.overlap_pct,
            populate_db=request.populate_db,
            offline_fallback=True
        )

        return {
            "status": "success",
            "region_id": result.region_id,
            "source_type": result.source_type,
            "total_tiles_generated": result.total_tiles_generated,
            "scenes_processed": [
                {
                    "scene_id": s.scene_id,
                    "time_bucket": s.time_bucket,
                    "acquisition_date": s.acquisition_date,
                    "tiles_count": s.tiles_count,
                    "cloud_pct": s.cloud_pct,
                    "manifest_path": s.manifest_path
                }
                for s in result.scenes_processed
            ],
            "elapsed_seconds": result.elapsed_seconds,
            "is_offline": result.is_offline
        }

    except Exception as e:
        logger.error(f"AOI Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AOI Ingestion failed: {str(e)}")


@router.post("/file", response_model=Dict[str, Any])
def ingest_file(request: FileIngestRequest):
    """Entry Point B: Ingests an evaluation GeoTIFF file directly (offline)."""
    try:
        target_path = request.file_paths if (request.file_paths and len(request.file_paths) > 0) else request.file_path
        if not target_path:
            raise ValueError("Either 'file_path' or 'file_paths' must be provided.")

        result: PipelineResult = run_direct_file_ingestion_pipeline(
            file_path=target_path,
            region_id=request.region_id,
            custom_band_order=request.custom_band_order,
            acquisition_date=request.acquisition_date,
            sensor=request.sensor or "auto",
            ground_crop_size=request.ground_crop_size,
            overlap_pct=request.overlap_pct,
            populate_db=request.populate_db
        )

        return {
            "status": "success",
            "region_id": result.region_id,
            "source_type": result.source_type,
            "total_tiles_generated": result.total_tiles_generated,
            "scenes_processed": [
                {
                    "scene_id": s.scene_id,
                    "time_bucket": s.time_bucket,
                    "acquisition_date": s.acquisition_date,
                    "tiles_count": s.tiles_count,
                    "cloud_pct": s.cloud_pct,
                    "manifest_path": s.manifest_path
                }
                for s in result.scenes_processed
            ],
            "elapsed_seconds": result.elapsed_seconds,
            "is_offline": result.is_offline
        }
    except Exception as e:
        logger.error(f"File Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"File Ingestion failed: {str(e)}")


@router.post("/upload", response_model=Dict[str, Any])
async def upload_and_ingest_file(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None),
    sensor: Optional[str] = Form("auto"),
    region_id: Optional[str] = Form(None),
    custom_band_order: Optional[str] = Form(None),
    acquisition_date: Optional[str] = Form(None)
):
    """Uploads local GeoTIFF file(s) and runs Entry Point B ingestion."""
    try:
        uploaded_list: List[UploadFile] = []
        if files:
            uploaded_list.extend(files)
        if file:
            uploaded_list.append(file)

        if not uploaded_list:
            raise HTTPException(status_code=400, detail="No files uploaded.")

        temp_dir = Path("data/uploads") / f"upload_{uuid.uuid4().hex[:8]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []

        for up_file in uploaded_list:
            if not up_file.filename:
                continue
            fname = Path(up_file.filename).name
            saved_file_path = temp_dir / fname
            with open(saved_file_path, "wb") as buffer:
                shutil.copyfileobj(up_file.file, buffer)
            saved_paths.append(saved_file_path)

        if not saved_paths:
            raise HTTPException(status_code=400, detail="Uploaded files were empty.")

        band_order_list = [b.strip() for b in custom_band_order.split(",")] if custom_band_order else None
        pipeline_input = saved_paths if len(saved_paths) > 1 else saved_paths[0]
        derived_region = region_id or f"upload_{saved_paths[0].stem}"

        result = run_direct_file_ingestion_pipeline(
            file_path=pipeline_input,
            region_id=derived_region,
            custom_band_order=band_order_list,
            acquisition_date=acquisition_date,
            sensor=sensor or "auto",
            populate_db=True
        )

        return {
            "status": "success",
            "region_id": result.region_id,
            "source_type": "file_upload",
            "files_uploaded": len(saved_paths),
            "total_tiles_generated": result.total_tiles_generated,
            "elapsed_seconds": result.elapsed_seconds
        }
    except Exception as e:
        logger.error(f"Upload Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Upload Ingestion failed: {str(e)}")


@router.post("/maxar", response_model=Dict[str, Any])
def ingest_maxar(request: MaxarIngestRequest):
    """
    Async Maxar WorldView / Wayback WMTS ingestion.
    Returns job_id INSTANTLY — no browser timeout, works for any AOI size.
    Poll GET /api/v1/ingest/job/{job_id} for real-time progress.
    """
    if is_offline_mode():
        raise HTTPException(
            status_code=403,
            detail="Canopus is currently operating in Air-Gapped Offline Mode. External Maxar WMTS requests are disabled. Switch to Online Mode in the system settings to fetch remote imagery."
        )

    derived_region = request.region_id or f"maxar_aoi_{uuid.uuid4().hex[:6]}"
    job_id = uuid.uuid4().hex[:12]

    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "status": "queued",
            "progress": 0,
            "message": "Job queued — pipeline starting...",
            "result": None,
            "region_id": derived_region,
            "created_at": datetime.utcnow().isoformat()
        }

    t = threading.Thread(
        target=_run_maxar_job,
        args=(job_id, derived_region, request.dict()),
        daemon=True
    )
    t.start()

    logger.info(f"Maxar async job {job_id} started for region '{derived_region}' epochs {request.years}")
    return {
        "status": "queued",
        "job_id": job_id,
        "region_id": derived_region,
        "message": f"Maxar ingestion started in background. Poll /api/v1/ingest/job/{job_id} for status."
    }


@router.get("/job/{job_id}", response_model=Dict[str, Any])
def get_job_status(job_id: str):
    """
    Poll a background Maxar ingestion job.
    Returns: status (queued|running|done|failed), progress (0-100), message, result when done.
    """
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)

    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found. It may have expired or the server restarted."
        )

    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "message": job["message"],
        "region_id": job.get("region_id"),
        "created_at": job.get("created_at"),
        "result": job.get("result")
    }
