"""
backend/api/routers/review.py
==============================
Unified Analyst Review & Verification Router

Endpoints:
  - Semantic Retrieval Auditing:
      * GET  /api/v1/review/retrieval/queries
      * POST /api/v1/review/retrieval/feedback
      * GET  /api/v1/review/retrieval/feedback/{query_text}
      * POST /api/v1/review/retrieval/rerank
      * GET  /api/v1/review/retrieval/export.geojson
  - Unified Audit Timeline:
      * GET  /api/v1/review/audit-log
"""

import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, Body

from backend.services.analyst_workflow import (
    get_recent_search_queries,
    get_query_detailed_results,
    record_search_feedback,
    get_search_query_feedback,
    apply_rocchio_rerank,
    export_retrieval_dossier_geojson,
    get_unified_audit_log
)

logger = logging.getLogger("review_router")
router = APIRouter(prefix="/api/v1/review", tags=["Analyst Review Workbench"])


class SearchFeedbackRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Query prompt text that produced the result")
    search_log_id: Optional[Any] = Field(None, description="Search log database ID")
    tile_id: str = Field(..., description="Target satellite tile ID being reviewed")
    feedback_type: Optional[str] = Field(None, description="'positive' or 'negative'")
    relevant: Optional[bool] = Field(None, description="True if relevant target, False if false alarm")
    relevance_score: float = Field(1.0, ge=0.0, le=1.0, description="Confidence/relevance rating")
    tag: str = Field("", description="Target classification tag e.g. 'airbase', 'runway'")
    note: str = Field("", description="Analyst rationale or notes")
    analyst_id: str = Field("ANALYST-DEF-01", description="Reviewing officer identifier")


class RocchioRerankRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Query prompt text to rerank")
    search_log_id: Optional[Any] = Field(None, description="Search log database ID")
    top_k: int = Field(10, ge=1, le=50, description="Top K items to retrieve")
    alpha: float = Field(1.0, description="Weight of original query vector")
    beta: float = Field(0.75, description="Weight of positive feedback centroid")
    gamma: float = Field(0.25, description="Weight of negative feedback centroid")


@router.get("/retrieval/queries", response_model=Dict[str, Any])
def list_audited_queries(
    limit: int = Query(50, ge=1, le=200),
    query_type: Optional[str] = Query(None, description="Filter by 'text' or 'image'")
):
    """
    Returns list of recently executed search queries and their feedback tallies.
    """
    try:
        all_queries = get_recent_search_queries(limit=100, query_type=None)
        text_count = sum(1 for q in all_queries if q.get("query_type") == "text")
        image_count = sum(1 for q in all_queries if q.get("query_type") == "image")

        if query_type:
            queries = [q for q in all_queries if q.get("query_type") == query_type][:limit]
        else:
            queries = all_queries[:limit]

        return {
            "status": "success",
            "total": len(queries),
            "text_count": text_count,
            "image_count": image_count,
            "queries": queries
        }
    except Exception as e:
        logger.error(f"Error fetching queries: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/feedback", response_model=Dict[str, Any])
def submit_search_feedback(req: SearchFeedbackRequest):
    """
    Submits analyst relevance feedback (thumbs up / thumbs down / tag / note) on a search result.
    """
    try:
        query_text = req.query_text
        search_id_int = int(req.search_log_id) if req.search_log_id is not None and str(req.search_log_id).isdigit() else None
        if search_id_int:
            from backend.services.analyst_workflow import get_pg_connection
            import psycopg2.extras
            conn = get_pg_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT raw_query, query_type FROM search_log WHERE search_id = %s;", (search_id_int,))
                    row = cur.fetchone()
                    if row and row.get("raw_query"):
                        query_text = row["raw_query"]
            finally:
                conn.close()

        if not query_text:
            query_text = str(search_id_int or "unknown_query")

        # Map feedback_type to boolean relevant if not explicitly set
        relevant = req.relevant
        if relevant is None:
            relevant = (req.feedback_type == "positive")

        feedback = record_search_feedback(
            query_text=query_text,
            tile_id=req.tile_id,
            relevant=relevant,
            relevance_score=req.relevance_score,
            tag=req.tag,
            note=req.note,
            analyst_id=req.analyst_id,
            search_id=search_id_int
        )
        return {"status": "success", "feedback": feedback}
    except Exception as e:
        logger.error(f"Error recording search feedback: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval/results", response_model=Dict[str, Any])
def get_query_results_enriched(
    query: Optional[str] = Query(None, description="Query text or image identifier"),
    search_log_id: Optional[str] = Query(None, description="Search log database ID"),
    query_type: Optional[str] = Query("text", description="Query type: 'text' or 'image'"),
    top_k: int = Query(24, ge=1, le=50, description="Top K tiles to return")
):
    """
    Returns search results for a query populated with their current analyst feedback state,
    including accepted, rejected, and pending counts.
    """
    try:
        data = get_query_detailed_results(query_text=query, query_type=query_type, search_log_id=search_log_id, top_k=top_k)
        return data
    except Exception as e:
        logger.error(f"Error getting detailed query results for '{query or search_log_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval/feedback", response_model=Dict[str, Any])
def get_query_feedback(query: str = Query(..., description="The query string to look up")):
    """
    Retrieves all analyst review annotations for a specific query text.
    """
    try:
        feedbacks = get_search_query_feedback(query_text=query)
        return {"status": "success", "query": query, "feedback_count": len(feedbacks), "feedbacks": feedbacks}
    except Exception as e:
        logger.error(f"Error fetching feedback for '{query}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval/rerank", response_model=Dict[str, Any])
def rerank_with_feedback(req: RocchioRerankRequest):
    """
    Executes Rocchio Relevance Feedback vector reranking using analyst ratings.
    """
    try:
        query_text = req.query_text
        if not query_text and req.search_log_id:
            from backend.services.analyst_workflow import get_pg_connection
            import psycopg2.extras
            conn = get_pg_connection()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("SELECT raw_query FROM search_log WHERE search_id = %s;", (int(req.search_log_id),))
                    row = cur.fetchone()
                    if row:
                        query_text = row["raw_query"]
            finally:
                conn.close()

        if not query_text:
            query_text = str(req.search_log_id or "")

        reranked = apply_rocchio_rerank(
            query_text=query_text,
            search_log_id=req.search_log_id,
            top_k=req.top_k,
            alpha=req.alpha,
            beta=req.beta,
            gamma=req.gamma
        )
        return reranked
    except Exception as e:
        logger.error(f"Error during Rocchio rerank: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/retrieval/export.geojson")
def export_target_dossier_geojson(query: str = Query(..., description="Query prompt text")):
    """
    Exports verified relevant targets for this query as a defense target dossier (GeoJSON).
    """
    try:
        return export_retrieval_dossier_geojson(query_text=query)
    except Exception as e:
        logger.error(f"Error exporting dossier GeoJSON: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit-log", response_model=Dict[str, Any])
def fetch_unified_audit_log(limit: int = Query(100, ge=1, le=500)):
    """
    Returns unified chronological audit trail across both Change Analysis and Semantic Retrieval.
    """
    try:
        logs = get_unified_audit_log(limit=limit)
        return {"status": "success", "total_records": len(logs), "audit_log": logs}
    except Exception as e:
        logger.error(f"Error fetching audit log: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
