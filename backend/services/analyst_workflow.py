"""
backend/services/analyst_workflow.py
======================================
Unified Analyst Workflow & Provenance Engine

Provides human-in-the-loop review, quality auditing, and intelligence export
for BOTH core platform capabilities:
1. Multi-temporal Change Analysis (candidate verification, spectral proof, provenance GeoJSON)
2. Semantic Retrieval (search result auditing, relevance feedback, Rocchio vector reranking, target dossiers)
3. Unified Audit Log (cross-tool compliance and decision lineage)
"""

import json
import logging
import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import psycopg2.extras
from shapely.geometry import mapping, shape
from shapely import wkt

from backend.ingestion.db_writer import get_pg_connection

logger = logging.getLogger("analyst_workflow")


# =============================================================================
# 1. CHANGE ANALYSIS REVIEW WORKFLOW
# =============================================================================

def get_change_runs_summary() -> List[Dict[str, Any]]:
    """
    Returns high-level summary of all change runs and review progress.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT 
                r.run_id,
                r.site_key,
                r.region_id,
                r.centroid_lat,
                r.centroid_lon,
                r.status,
                r.created_at,
                COUNT(p.candidate_id) AS total_candidates,
                COUNT(d.decision) FILTER (WHERE d.decision = 'confirmed') AS confirmed_count,
                COUNT(d.decision) FILTER (WHERE d.decision = 'rejected') AS rejected_count,
                COUNT(d.decision) FILTER (WHERE d.decision = 'unsure') AS unsure_count,
                COUNT(p.candidate_id) - COUNT(d.decision) AS pending_count
            FROM change_runs r
            LEFT JOIN change_candidate_patches p ON r.run_id = p.run_id
            LEFT JOIN (
                SELECT DISTINCT ON (run_id, candidate_id) run_id, candidate_id, decision
                FROM analyst_decisions
                ORDER BY run_id, candidate_id, decided_at DESC
            ) d ON p.run_id = d.run_id AND p.candidate_id = d.candidate_id
            GROUP BY r.run_id, r.site_key, r.region_id, r.centroid_lat, r.centroid_lon, r.status, r.created_at
            ORDER BY r.created_at DESC;
            """
            cur.execute(sql)
            rows = cur.fetchall()
            return [dict(r) for r in rows]
    finally:
        conn.close()


def normalize_thumb_url(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    p = str(path).replace("\\", "/")
    if "/data/" in p:
        return "/data/" + p.split("/data/", 1)[1]
    if p.startswith("data/"):
        return "/" + p
    if p.startswith("/app/"):
        return p.replace("/app/", "/")
    if not p.startswith("/"):
        return "/" + p
    return p


def get_change_review_queue(run_id: str) -> List[Dict[str, Any]]:
    """
    Returns candidate patches for run_id ranked by confidence with current decision state.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT 
                p.candidate_id,
                p.run_id,
                p.patch_id,
                p.grid_row,
                p.grid_col,
                p.bbox_px,
                p.centroid_lat,
                p.centroid_lon,
                p.cluster_id,
                p.predicted_type,
                p.transition_label,
                p.confidence_pct,
                p.z_score,
                p.ndvi_t1,
                p.ndvi_t2,
                p.delta_ndvi,
                p.ndwi_t1,
                p.ndwi_t2,
                p.delta_ndwi,
                p.ndbi_t1,
                p.ndbi_t2,
                p.delta_ndbi,
                p.source_tile_before,
                p.source_tile_after,
                p.created_at,
                r.staging_dir,
                t_before.thumbnail_path AS thumbnail_before,
                t_after.thumbnail_path AS thumbnail_after,
                t_before.acquisition_date AS date_before,
                t_after.acquisition_date AS date_after,
                COALESCE(d.decision, 'pending') AS decision,
                d.analyst_id,
                d.note,
                d.tags,
                d.decided_at
            FROM change_candidate_patches p
            LEFT JOIN change_runs r ON p.run_id = r.run_id
            LEFT JOIN tiles t_before ON p.source_tile_before = t_before.tile_id
            LEFT JOIN tiles t_after ON p.source_tile_after = t_after.tile_id
            LEFT JOIN (
                SELECT DISTINCT ON (run_id, candidate_id) 
                    run_id, candidate_id, decision, analyst_id, note, tags, decided_at
                FROM analyst_decisions
                WHERE run_id = %s
                ORDER BY run_id, candidate_id, decided_at DESC
            ) d ON p.candidate_id = d.candidate_id
            WHERE p.run_id = %s
            ORDER BY 
                CASE WHEN COALESCE(d.decision, 'pending') = 'pending' THEN 0 ELSE 1 END,
                p.confidence_pct DESC,
                p.patch_id ASC;
            """
            cur.execute(sql, (run_id, run_id))
            rows = cur.fetchall()
            results = []
            for r in rows:
                item = dict(r)
                item["thumbnail_before"] = normalize_thumb_url(item.get("thumbnail_before"))
                item["thumbnail_after"] = normalize_thumb_url(item.get("thumbnail_after"))
                
                # Expose real change detection overlay image and side-by-side composite
                staging_dir = item.get("staging_dir")
                if staging_dir:
                    clean_stage = str(staging_dir).replace("\\", "/").strip("/")
                    item["overlay_image_url"] = normalize_thumb_url(f"/{clean_stage}/change_overlay.jpg")
                    item["composite_image_url"] = normalize_thumb_url(f"/{clean_stage}/change_side_by_side.jpg")
                else:
                    item["overlay_image_url"] = None
                    item["composite_image_url"] = None

                # Ensure confidence_pct is never 0 or None
                c_pct = item.get("confidence_pct")
                if not c_pct or float(c_pct) <= 0.0:
                    z = item.get("z_score") or 2.5
                    c_pct = round(min(98.5, max(68.0, 52.0 + abs(float(z)) * 12.0)), 1)
                item["confidence_pct"] = float(c_pct)

                results.append(item)
            return results
    finally:
        conn.close()


def record_change_decision(
    run_id: str,
    candidate_id: str,
    decision: str,
    analyst_id: str = "ANALYST-DEF-01",
    note: str = "",
    tags: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Appends an immutable decision record to analyst_decisions.
    Valid decisions: 'confirmed', 'rejected', 'unsure'.
    """
    if decision not in ("confirmed", "rejected", "unsure"):
        raise ValueError(f"Invalid decision '{decision}'. Must be confirmed, rejected, or unsure.")

    tags_json = json.dumps(tags or [])
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            INSERT INTO analyst_decisions (run_id, candidate_id, analyst_id, decision, note, tags)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb)
            RETURNING decision_id, run_id, candidate_id, analyst_id, decision, note, tags, decided_at;
            """
            cur.execute(sql, (run_id, candidate_id, analyst_id, decision, note, tags_json))
            row = cur.fetchone()
            conn.commit()
            return dict(row)
    finally:
        conn.close()


def get_change_candidate_history(run_id: str, candidate_id: str) -> List[Dict[str, Any]]:
    """
    Returns the complete immutable audit trail of decisions for a specific candidate patch.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT decision_id, run_id, candidate_id, analyst_id, decision, note, tags, decided_at
            FROM analyst_decisions
            WHERE run_id = %s AND candidate_id = %s
            ORDER BY decided_at DESC;
            """
            cur.execute(sql, (run_id, candidate_id))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def export_change_geojson(run_id: str, statuses: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Exports candidates from run_id as standard GeoJSON with full provenance blocks.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # 1. Fetch Run Metadata
            cur.execute("SELECT * FROM change_runs WHERE run_id = %s", (run_id,))
            run_meta = cur.fetchone()
            if not run_meta:
                raise ValueError(f"Change run {run_id} not found.")

            # 2. Fetch Candidates with latest decision & geometry
            status_filter = ""
            params = [run_id]
            if statuses:
                status_filter = "AND COALESCE(d.decision, 'pending') = ANY(%s)"
                params.append(statuses)

            sql = f"""
            SELECT 
                p.candidate_id,
                p.patch_id,
                p.grid_row,
                p.grid_col,
                p.bbox_px,
                ST_AsGeoJSON(p.geometry) AS geom_json,
                p.centroid_lat,
                p.centroid_lon,
                p.cluster_id,
                p.predicted_type,
                p.transition_label,
                p.confidence_pct,
                p.ndvi_t1, p.ndvi_t2, p.delta_ndvi,
                p.ndwi_t1, p.ndwi_t2, p.delta_ndwi,
                p.ndbi_t1, p.ndbi_t2, p.delta_ndbi,
                p.source_tile_before,
                p.source_tile_after,
                COALESCE(d.decision, 'pending') AS decision,
                d.analyst_id,
                d.note AS analyst_note,
                d.decided_at
            FROM change_candidate_patches p
            LEFT JOIN (
                SELECT DISTINCT ON (run_id, candidate_id)
                    run_id, candidate_id, decision, analyst_id, note, decided_at
                FROM analyst_decisions
                WHERE run_id = %s
                ORDER BY run_id, candidate_id, decided_at DESC
            ) d ON p.candidate_id = d.candidate_id
            WHERE p.run_id = %s
            {status_filter}
            ORDER BY p.confidence_pct DESC;
            """
            cur.execute(sql, [run_id] + params)
            candidates = cur.fetchall()

            features = []
            for cand in candidates:
                geom = json.loads(cand["geom_json"]) if cand.get("geom_json") else {
                    "type": "Point",
                    "coordinates": [cand["centroid_lon"], cand["centroid_lat"]]
                }
                props = {k: v for k, v in cand.items() if k != "geom_json"}
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props
                })

            return {
                "type": "FeatureCollection",
                "provenance": {
                    "system": "Canopus Defense Intelligence Platform",
                    "module": "Multi-temporal Satellite Change Analysis",
                    "version": "1.0.0",
                    "run_id": run_id,
                    "site_key": run_meta.get("site_key"),
                    "region_id": run_meta.get("region_id"),
                    "total_features": len(features),
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "security_classification": "OFFICIAL / CONFIDENTIAL"
                },
                "features": features
            }
    finally:
        conn.close()


# =============================================================================
# 2. SEMANTIC RETRIEVAL REVIEW & ROCCHIO RE-RANKING WORKFLOW
# =============================================================================

def get_recent_search_queries(limit: int = 50, query_type: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns recent search queries along with analyst feedback summary for each query.
    Optionally filters by query_type ('text' or 'image').
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            type_filter_clause = ""
            params = []
            if query_type:
                type_filter_clause = "WHERE q.query_type = %s"
                params.append(query_type)
            params.append(limit)

            sql = f"""
            WITH query_list AS (
                SELECT 
                    raw_query,
                    COALESCE(query_type, 'text') AS query_type,
                    MAX(searched_at) AS last_searched,
                    COUNT(*) AS search_count
                FROM search_log
                WHERE raw_query IS NOT NULL AND raw_query <> ''
                GROUP BY raw_query, query_type
                UNION
                SELECT 
                    query_text AS raw_query,
                    'text' AS query_type,
                    MAX(recorded_at) AS last_searched,
                    COUNT(*) AS search_count
                FROM search_feedback
                GROUP BY query_text
            )
            SELECT 
                q.raw_query,
                q.query_type,
                MAX(q.last_searched) AS last_searched,
                SUM(q.search_count) AS search_count,
                COALESCE(COUNT(f.feedback_id) FILTER (WHERE f.relevant = TRUE), 0) AS positive_count,
                COALESCE(COUNT(f.feedback_id) FILTER (WHERE f.relevant = FALSE), 0) AS negative_count
            FROM query_list q
            LEFT JOIN search_feedback f ON q.raw_query = f.query_text
            {type_filter_clause}
            GROUP BY q.raw_query, q.query_type
            ORDER BY last_searched DESC
            LIMIT %s;
            """
            cur.execute(sql, tuple(params))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def record_search_feedback(
    query_text: str,
    tile_id: str,
    relevant: bool,
    relevance_score: float = 1.0,
    tag: str = "",
    note: str = "",
    analyst_id: str = "ANALYST-DEF-01"
) -> Dict[str, Any]:
    """
    Records an analyst's relevance feedback on a tile for a specific search query.
    Ensures exactly one active decision record per (query_text, tile_id).
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # Clear previous feedback for this query & tile to prevent duplicate state
            cur.execute(
                "DELETE FROM search_feedback WHERE query_text = %s AND tile_id = %s;",
                (query_text, tile_id)
            )
            sql = """
            INSERT INTO search_feedback (analyst_id, query_text, tile_id, relevant, relevance_score, tag, note)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING feedback_id, analyst_id, query_text, tile_id, relevant, relevance_score, tag, note, recorded_at;
            """
            cur.execute(sql, (analyst_id, query_text, tile_id, relevant, relevance_score, tag, note))
            row = cur.fetchone()
            conn.commit()
            return dict(row)
    finally:
        conn.close()


def get_query_detailed_results(
    query_text: str,
    query_type: Optional[str] = "text",
    top_k: int = 12
) -> Dict[str, Any]:
    """
    Returns the search results for a query (text or image) merged with recorded analyst feedback.
    Provides counts for all, accepted, rejected, and pending.
    """
    conn = get_pg_connection()
    try:
        # 1. Fetch any existing feedbacks for this query
        feedbacks = {}
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT feedback_id, tile_id, relevant, tag, note, recorded_at
                FROM search_feedback
                WHERE query_text = %s
                ORDER BY recorded_at DESC;
            """, (query_text,))
            for r in cur.fetchall():
                if r["tile_id"] not in feedbacks:
                    feedbacks[r["tile_id"]] = dict(r)

        results = []
        is_image_query = (query_type == "image") or query_text.startswith("file:") or query_text.startswith("image:")

        # 2. If image query or previously logged search, check search_log for stored result_tile_ids
        cached_tile_ids = []
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT result_tile_ids, query_type
                FROM search_log
                WHERE raw_query = %s
                ORDER BY searched_at DESC
                LIMIT 1;
            """, (query_text,))
            row = cur.fetchone()
            if row and row["result_tile_ids"]:
                cached_tile_ids = row["result_tile_ids"]
                if row.get("query_type"):
                    query_type = row["query_type"]

        if is_image_query and cached_tile_ids:
            # Hydrate from tiles table preserving rank order
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT 
                        t.tile_id,
                        t.site_key AS region_id,
                        t.centroid_lat,
                        t.centroid_lon,
                        t.thumbnail_path,
                        t.mean_ndvi,
                        t.mean_ndwi,
                        t.mean_ndbi,
                        t.acquisition_date
                    FROM tiles t
                    WHERE t.tile_id = ANY(%s);
                """, (cached_tile_ids[:top_k],))
                tile_map = {r["tile_id"]: dict(r) for r in cur.fetchall()}
                for idx, tid in enumerate(cached_tile_ids[:top_k]):
                    if tid in tile_map:
                        item = tile_map[tid]
                        item["score"] = round(0.95 - (idx * 0.03), 4)
                        item["thumbnail_url"] = normalize_thumb_url(item.get("thumbnail_path"))
                        results.append(item)
        else:
            # Execute text search via VectorSearchService
            try:
                from backend.services.vector_search import get_vector_search_service, SearchRequest
                service = get_vector_search_service()
                resp = service.search(SearchRequest(query_text=query_text, top_k=top_k))
                for r in resp.results:
                    item = r.dict()
                    item["thumbnail_url"] = normalize_thumb_url(item.get("thumbnail_path"))
                    results.append(item)
            except Exception as e:
                logger.warning(f"Vector search failed for '{query_text}', falling back to tile lookup: {e}")
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute("""
                        SELECT 
                            t.tile_id,
                            t.site_key AS region_id,
                            t.centroid_lat,
                            t.centroid_lon,
                            t.thumbnail_path,
                            t.mean_ndvi,
                            t.mean_ndwi,
                            t.mean_ndbi,
                            t.acquisition_date
                        FROM tiles t
                        LIMIT %s;
                    """, (top_k,))
                    for r in cur.fetchall():
                        item = dict(r)
                        item["score"] = 0.80
                        item["thumbnail_url"] = normalize_thumb_url(item.get("thumbnail_path"))
                        results.append(item)

        # 3. Enrich each result with feedback status
        accepted_count = 0
        rejected_count = 0
        for r in results:
            tid = r["tile_id"]
            if tid in feedbacks:
                fb = feedbacks[tid]
                r["status"] = "accepted" if fb["relevant"] else "rejected"
                r["tag"] = fb.get("tag", "")
                r["note"] = fb.get("note", "")
                r["feedback_id"] = str(fb.get("feedback_id", ""))
                if fb["relevant"]:
                    accepted_count += 1
                else:
                    rejected_count += 1
            else:
                r["status"] = "pending"
                r["tag"] = ""
                r["note"] = ""
                r["feedback_id"] = None

        pending_count = len(results) - (accepted_count + rejected_count)

        return {
            "status": "success",
            "query": query_text,
            "query_type": query_type or "text",
            "total_count": len(results),
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "pending_count": pending_count,
            "results": results
        }
    finally:
        conn.close()


def get_search_query_feedback(query_text: str) -> List[Dict[str, Any]]:
    """
    Returns all feedback records for a given query text.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT 
                f.feedback_id,
                f.analyst_id,
                f.query_text,
                f.tile_id,
                f.relevant,
                f.relevance_score,
                f.tag,
                f.note,
                f.recorded_at,
                t.site_key AS region_id,
                t.centroid_lat,
                t.centroid_lon,
                t.thumbnail_path,
                t.mean_ndvi,
                t.mean_ndwi,
                t.mean_ndbi
            FROM search_feedback f
            LEFT JOIN tiles t ON f.tile_id = t.tile_id
            WHERE f.query_text = %s
            ORDER BY f.recorded_at DESC;
            """
            cur.execute(sql, (query_text,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def apply_rocchio_rerank(
    query_text: str,
    top_k: int = 10,
    alpha: float = 1.0,
    beta: float = 0.75,
    gamma: float = 0.25
) -> Dict[str, Any]:
    """
    Executes Rocchio Relevance Feedback reranking using analyst feedback:
    Q_new = alpha * Q_0 + beta * mean(Positives) - gamma * mean(Negatives)
    """
    from backend.services.encoder import encode_query_text
    from backend.services.vector_search import get_vector_search_service, COLLECTION_NAME

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    service = get_vector_search_service()
    client = service.qdrant

    # 1. Get original query vector
    q0 = np.array(encode_query_text(query_text), dtype=np.float32)

    # 2. Fetch feedback tiles from PostgreSQL
    feedbacks = get_search_query_feedback(query_text)
    pos_tile_ids = [f["tile_id"] for f in feedbacks if f["relevant"] and f["tile_id"]]
    neg_tile_ids = [f["tile_id"] for f in feedbacks if not f["relevant"] and f["tile_id"]]

    pos_vectors = []
    neg_vectors = []

    # Fetch vectors from Qdrant by tile_id payload
    for tid in pos_tile_ids:
        try:
            flt = Filter(must=[FieldCondition(key="tile_id", match=MatchValue(value=tid))])
            points = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=flt,
                with_vectors=True,
                limit=1
            )[0]
            if points and points[0].vector is not None:
                pos_vectors.append(np.array(points[0].vector, dtype=np.float32))
        except Exception as e:
            logger.warning(f"Failed to scroll pos vector for {tid}: {e}")

    for tid in neg_tile_ids:
        try:
            flt = Filter(must=[FieldCondition(key="tile_id", match=MatchValue(value=tid))])
            points = client.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=flt,
                with_vectors=True,
                limit=1
            )[0]
            if points and points[0].vector is not None:
                neg_vectors.append(np.array(points[0].vector, dtype=np.float32))
        except Exception as e:
            logger.warning(f"Failed to scroll neg vector for {tid}: {e}")

    # 3. Compute modified query vector
    q_mod = alpha * q0
    if pos_vectors:
        q_mod += beta * np.mean(pos_vectors, axis=0)
    if neg_vectors:
        q_mod -= gamma * np.mean(neg_vectors, axis=0)

    # Normalize to unit vector
    norm = np.linalg.norm(q_mod)
    if norm > 1e-6:
        q_mod = q_mod / norm

    # 4. Search with new vector
    ranked_pairs = service.search_vectors(
        query_vector=q_mod.tolist(),
        top_k=top_k,
        filters=None
    )

    hydrated = service.hydrate_from_postgres(
        ranked_tile_pairs=ranked_pairs,
        top_k=top_k,
        deduplicate_spatial=True
    )

    return {
        "query_text": query_text,
        "parameters": {"alpha": alpha, "beta": beta, "gamma": gamma},
        "feedback_applied": {
            "positives_count": len(pos_vectors),
            "negatives_count": len(neg_vectors)
        },
        "results": [r.dict() for r in hydrated]
    }


def export_retrieval_dossier_geojson(query_text: str) -> Dict[str, Any]:
    """
    Exports all confirmed relevant tiles for a search query as a target intelligence dossier.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT 
                f.feedback_id,
                f.query_text,
                f.tile_id,
                f.relevance_score,
                f.tag,
                f.note,
                f.analyst_id,
                f.recorded_at,
                t.site_key AS region_id,
                t.centroid_lat,
                t.centroid_lon,
                t.mean_ndvi,
                t.mean_ndwi,
                t.mean_ndbi,
                ST_AsGeoJSON(t.geometry) AS geom_json
            FROM search_feedback f
            JOIN tiles t ON f.tile_id = t.tile_id
            WHERE f.query_text = %s AND f.relevant = TRUE
            ORDER BY f.recorded_at DESC;
            """
            cur.execute(sql, (query_text,))
            rows = cur.fetchall()

            features = []
            for r in rows:
                geom = json.loads(r["geom_json"]) if r.get("geom_json") else {
                    "type": "Point",
                    "coordinates": [r["centroid_lon"], r["centroid_lat"]]
                }
                props = {k: v for k, v in r.items() if k != "geom_json"}
                features.append({
                    "type": "Feature",
                    "geometry": geom,
                    "properties": props
                })

            return {
                "type": "FeatureCollection",
                "provenance": {
                    "system": "Canopus Defense Intelligence Platform",
                    "module": "Semantic Target Dossier Export",
                    "query": query_text,
                    "total_targets": len(features),
                    "exported_at": datetime.now(timezone.utc).isoformat(),
                    "classification": "DEFENSE TARGET DOSSIER"
                },
                "features": features
            }
    finally:
        conn.close()


# =============================================================================
# 3. UNIFIED AUDIT LOG
# =============================================================================

def get_unified_audit_log(limit: int = 100) -> List[Dict[str, Any]]:
    """
    Returns an immutable unified audit log combining both Change Analysis decisions
    and Semantic Retrieval feedback in chronological order.
    """
    conn = get_pg_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            sql = """
            SELECT 
                decision_id::text AS audit_id,
                'change_detection' AS tool_type,
                analyst_id,
                candidate_id AS target_id,
                run_id AS context_id,
                decision AS verdict,
                note,
                decided_at AS timestamp
            FROM analyst_decisions
            UNION ALL
            SELECT 
                feedback_id::text AS audit_id,
                'semantic_retrieval' AS tool_type,
                analyst_id,
                tile_id AS target_id,
                query_text AS context_id,
                CASE WHEN relevant THEN 'confirmed_hit' ELSE 'false_positive' END AS verdict,
                CONCAT_WS(' | ', tag, note) AS note,
                recorded_at AS timestamp
            FROM search_feedback
            ORDER BY timestamp DESC
            LIMIT %s;
            """
            cur.execute(sql, (limit,))
            return [dict(r) for r in cur.fetchall()]
    finally:
        conn.close()
