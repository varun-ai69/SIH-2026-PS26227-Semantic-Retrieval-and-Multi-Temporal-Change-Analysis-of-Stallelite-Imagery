"""
backend/api/routers/search.py
==============================
Phase 2.6: FastAPI Router for Multimodal Semantic Retrieval
============================================================
PS Sections: 2.2.1 (Semantic & Multimodal Image Retrieval)

Endpoints:
  - POST /api/v1/search (JSON payload for text prompt & metadata filters)
  - POST /api/v1/search/image (Multipart upload for reference image query & filters)
"""

import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Body

from backend.services.vector_search import (
    SearchRequest,
    SearchFilter,
    SearchResponse,
    get_vector_search_service
)

log = logging.getLogger("SearchRouter")

router = APIRouter(prefix="/api/v1/search", tags=["Semantic Retrieval"])


@router.post("", response_model=SearchResponse, summary="Text or JSON Semantic Vector Search")
def search_semantic_text(
    request: SearchRequest = Body(...)
):
    """
    Phase 2.6: Execute natural language semantic query across satellite tile archive.
    Supports optional spatial AOI polygon, date-range, sensor, and quality filters.
    """
    try:
        service = get_vector_search_service()
        response = service.search(request)
        return response
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal search error: {str(e)}")


@router.post("/image", response_model=SearchResponse, summary="Multimodal Image-to-Image Search")
async def search_semantic_image(
    file: UploadFile = File(..., description="Query reference satellite image (JPG, PNG, or GeoTIFF)"),
    top_k: int = Form(5, description="Number of top results to retrieve"),
    sensor: Optional[str] = Form(None, description="Optional sensor filter, e.g. 'Sentinel-2'"),
    start_date: Optional[str] = Form(None, description="Optional ISO start date (e.g. 2023-01-01)"),
    end_date: Optional[str] = Form(None, description="Optional ISO end date (e.g. 2024-01-01)"),
    min_quality: float = Form(0.0, description="Minimum quality gate score (0.0 - 1.0)"),
    max_cloud_pct: float = Form(100.0, description="Maximum cloud coverage percentage"),
    min_similarity: float = Form(0.65, description="Minimum similarity percentage cutoff (0.0 to 1.0, e.g. 0.65 for 65%)"),
    aoi_geojson: Optional[str] = Form(None, description="Optional GeoJSON Polygon string for spatial filtering"),
    analyst_id: str = Form("demo_analyst", description="Analyst identity for audit logging")
):
    """
    Phase 2.6: Execute image-to-image visual vector similarity query using uploaded file.
    """
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Parse filters
        aoi_poly = None
        if aoi_geojson:
            try:
                aoi_poly = json.loads(aoi_geojson)
            except Exception:
                pass

        dt_start = datetime.fromisoformat(start_date) if start_date else None
        dt_end = datetime.fromisoformat(end_date) if end_date else None

        search_filters = SearchFilter(
            sensor=sensor,
            start_date=dt_start,
            end_date=dt_end,
            min_quality=min_quality,
            max_cloud_pct=max_cloud_pct,
            aoi_polygon=aoi_poly
        )

        request = SearchRequest(
            query_image_bytes=contents,
            query_filename=file.filename,
            filters=search_filters,
            top_k=top_k,
            analyst_id=analyst_id,
            min_similarity=min_similarity
        )

        service = get_vector_search_service()
        response = service.search(request)
        return response

    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Image search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")
