"""
backend/services/vector_search.py
==================================
Phase 2.1 – 2.5: Semantic Vector Search Service & Filtered Query Engine
========================================================================
PS Sections: 2.2.1 (Semantic & Multimodal Retrieval), 2.2.6 (Scale & Sovereignty)

Provides:
  1. Phase 2.1: Reusable Query Encoding functions (`encode_query_text`, `encode_query_image`).
  2. Phase 2.2: Parameterized Vector Search with spatial polygon pre-filter,
     temporal date-range filter, sensor filter, and quality confidence gates.
  3. Phase 2.3: Batch Postgres Result Hydration preserving Qdrant similarity rank.
  4. Phase 2.4: Rule-based 3-Index Spot Description Generator (NDVI, NDWI, NDBI).
  5. Phase 2.5: Unified Agent-Ready SemanticSearchTool with `search_log` audit persistence.
"""

import os
import sys
import time
import json
import logging
from typing import Optional, List, Dict, Any, Union, Tuple
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
)

from backend.services.encoder import get_encoder, encode_query_text, encode_query_image
from backend.services.calibrator import get_calibrator

log = logging.getLogger("VectorSearch")

# Configuration & Defaults
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "tile_embeddings")

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5434")
PG_DB = os.getenv("POSTGRES_DB", "eo_archive")
PG_USER = os.getenv("POSTGRES_USER", "eo_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "eo_password")

PG_DSN = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"


# ============================================================
# Phase 2.4 — Comprehensive Multi-Index Rule-Based Explanation Engine
# ============================================================

def generate_tile_description(
    mean_ndvi: Optional[float] = None,
    mean_ndwi: Optional[float] = None,
    mean_ndbi: Optional[float] = None,
    sensor: Optional[str] = None
) -> str:
    """
    Phase 2.4 & 2.7: Comprehensive rule-based terrain and tactical spot explanation.
    Evaluates compound multi-spectral index pairs (NDVI, NDWI, NDBI) or high-res optical signatures.
    """
    if sensor and "maxar" in sensor.lower():
        vari_clause = f" (VARI visible vegetation index: {mean_ndvi:.2f})" if mean_ndvi is not None else ""
        return f"High-resolution sub-meter reconnaissance imagery (True Color optical). Optimized for detailed tactical object, vehicle, and infrastructure identification{vari_clause}."

    if mean_ndvi is None and mean_ndwi is None and mean_ndbi is None:
        return "Spectral indices unavailable for this ground tile."

    ndvi = mean_ndvi if mean_ndvi is not None else 0.0
    ndwi = mean_ndwi if mean_ndwi is not None else -0.2
    ndbi = mean_ndbi if mean_ndbi is not None else 0.0

    # 1. Holistic Landscape Inference Rules
    landscape_summary = ""

    if ndwi >= 0.25:
        landscape_summary = "Open surface aquatic body or major river channel with high specular water reflectance and minimal terrestrial signature."
    elif ndwi >= 0.05 and ndvi >= 0.20:
        landscape_summary = "Riparian wetland or active riverbank corridor exhibiting rich soil moisture and water-tolerant flora."
    elif ndbi >= 0.03 and ndvi < 0.18 and ndwi < -0.15:
        landscape_summary = "High-density paved industrial or transport corridor (e.g. runways, logistics yards, tarmac aprons) with minimal vegetation."
    elif ndbi >= 0.04 and ndvi < 0.24:
        landscape_summary = "Dense urban core or heavy commercial development characterized by concrete infrastructure, built structures, and road networks."
    elif ndbi >= 0.01 and 0.20 <= ndvi < 0.38:
        landscape_summary = "Mixed suburban landscape combining residential/commercial building clusters with planned green spaces and tree canopy."
    elif ndvi >= 0.40 and ndbi < -0.05:
        landscape_summary = "Dense natural woodland or thick forest canopy with high chlorophyll biomass and zero artificial built-up footprint."
    elif 0.26 <= ndvi < 0.40 and ndbi <= 0.02:
        landscape_summary = "Cultivated agricultural cropland or active vegetative pasture with prominent chlorophyll activity."
    elif ndvi < 0.16 and ndwi < -0.20 and ndbi < 0.03:
        landscape_summary = "Arid barren land or open cleared terrain with dry bare soil and sparse scrub, suitable for vehicle traversal or development."
    elif 0.16 <= ndvi < 0.26 and ndbi <= 0.02:
        landscape_summary = "Semi-arid open scrubland or sparse natural grassland with moderate background soil reflectance."
    else:
        # Balanced mixed terrain fallback
        landscape_summary = "Mixed transition terrain with balanced vegetation, soil, and low-density structural features."

    # 2. Quantitative Spectral Indices Breakdown
    # Vegetation clause
    if mean_ndvi is not None:
        if mean_ndvi >= 0.35:
            v_desc = f"dense vegetation (NDVI {mean_ndvi:.2f})"
        elif mean_ndvi >= 0.18:
            v_desc = f"sparse vegetation (NDVI {mean_ndvi:.2f})"
        else:
            v_desc = f"minimal vegetation (NDVI {mean_ndvi:.2f})"
    else:
        v_desc = None

    # Water clause
    if mean_ndwi is not None:
        if mean_ndwi >= 0.20:
            w_desc = f"prominent water presence (NDWI {mean_ndwi:.2f})"
        elif mean_ndwi >= -0.05:
            w_desc = f"moderate moisture (NDWI {mean_ndwi:.2f})"
        else:
            w_desc = f"dry surface moisture (NDWI {mean_ndwi:.2f})"
    else:
        w_desc = None

    # Built-up clause
    if mean_ndbi is not None:
        if mean_ndbi >= 0.04:
            b_desc = f"high built-up dominance (NDBI {mean_ndbi:.2f})"
        elif mean_ndbi >= 0.0:
            b_desc = f"moderate structural reflectance (NDBI {mean_ndbi:.2f})"
        else:
            b_desc = f"non-built natural surface (NDBI {mean_ndbi:.2f})"
    else:
        b_desc = None

    metric_parts = [p for p in [v_desc, w_desc, b_desc] if p]
    if metric_parts:
        metrics_sentence = "Spectral signature reflects " + ", ".join(metric_parts) + "."
    else:
        metrics_sentence = ""

    return f"{landscape_summary} {metrics_sentence}".strip()


# ============================================================
# 1. Pydantic Models for Search Contracts
# ============================================================

class SearchFilter(BaseModel):
    sensor: Optional[str] = Field(None, description="Filter by sensor, e.g. 'Sentinel-2'")
    start_date: Optional[datetime] = Field(None, description="Start date (UTC) for acquisition filtering")
    end_date: Optional[datetime] = Field(None, description="End date (UTC) for acquisition filtering")
    min_quality: Optional[float] = Field(0.0, ge=0.0, le=1.0, description="Minimum quality score (0.0 to 1.0)")
    max_cloud_pct: Optional[float] = Field(100.0, ge=0.0, le=100.0, description="Maximum cloud cover percentage (0-100)")
    aoi_polygon: Optional[Dict[str, Any]] = Field(
        None,
        description="GeoJSON Polygon / MultiPolygon dictionary for spatial intersection filtering"
    )
    bbox: Optional[List[float]] = Field(
        None,
        description="Bounding box [min_lon, min_lat, max_lon, max_lat] in EPSG:4326"
    )


class SearchRequest(BaseModel):
    query_text: Optional[str] = Field(None, description="Natural language prompt query")
    query_image_path: Optional[str] = Field(None, description="Path to query image file")
    query_image_bytes: Optional[bytes] = Field(None, description="Raw query image bytes")
    query_filename: Optional[str] = Field(None, description="Original uploaded filename for self-match exclusion")
    filters: Optional[SearchFilter] = Field(default_factory=SearchFilter)
    top_k: int = Field(5, ge=1, le=100, description="Number of top ranked results to retrieve")
    analyst_id: str = Field("demo_analyst", description="Analyst identity for audit logging")
    min_similarity: Optional[float] = Field(None, description="Minimum similarity threshold (0.0 to 1.0, e.g. 0.65 for 65%)")
    calibrate: bool = Field(True, description="Enable QB-Norm DIS calibration using 512 RSICD Query Bank")
    calibration_mode: str = Field("auto", description="'auto' (gated if top-1 is hub), 'always', or 'off'")


class SearchResultItem(BaseModel):
    tile_id: str
    score: float = Field(..., description="Calibrated similarity score (0.0 to 1.0)")
    raw_score: Optional[float] = Field(None, description="Original raw cosine similarity before calibration")
    is_hub: Optional[bool] = Field(False, description="Whether this candidate was identified as a generic hub")
    calibrated: Optional[bool] = Field(False, description="Whether QB-Norm score calibration was applied")
    scene_id: Optional[str] = None
    site_key: Optional[str] = None
    acquisition_date: Optional[str] = None
    centroid_lat: Optional[float] = None
    centroid_lon: Optional[float] = None
    sensor: Optional[str] = None
    cloud_pct: Optional[float] = 0.0
    quality_confidence: Optional[float] = 1.0
    mean_ndvi: Optional[float] = None
    mean_ndwi: Optional[float] = None
    mean_ndbi: Optional[float] = None
    spot_description: Optional[str] = None
    file_path: Optional[str] = None
    thumbnail_url: Optional[str] = None
    thumbnail_path: Optional[str] = None
    geometry_geojson: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    search_log_id: Optional[str] = Field(default=None, description="Database UUID of logged search session")
    query_type: str = Field(..., description="'text' or 'image'")
    query: str = Field(..., description="Summary representation of search query")
    total_found: int = Field(..., description="Number of results returned")
    filters_applied: Dict[str, Any] = Field(default_factory=dict)
    results: List[SearchResultItem] = Field(default_factory=list)
    execution_time_ms: float = Field(..., description="Total execution latency in milliseconds")


# ============================================================
# 2. VectorSearchService Implementation
# ============================================================

class VectorSearchService:
    """
    Core vector retrieval & enrichment service implementing Phase 2.1 – 2.5.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.encoder = get_encoder()
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            self.qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DSN)

    def resolve_aoi_to_tile_ids(
        self,
        aoi_polygon: Optional[Dict[str, Any]] = None,
        bbox: Optional[List[float]] = None
    ) -> Optional[List[str]]:
        """
        Phase 2.2 Task 3: Pre-resolves spatial AOI to a list of intersecting tile_ids
        via PostGIS ST_Intersects, then returns the candidate tile IDs.
        """
        if not aoi_polygon and not bbox:
            return None

        try:
            conn = self._get_pg_conn()
            with conn.cursor() as cur:
                if aoi_polygon:
                    geojson_str = json.dumps(aoi_polygon)
                    sql = """
                    SELECT tile_id FROM tiles
                    WHERE ST_Intersects(geometry, ST_SetSRID(ST_GeomFromGeoJSON(%s), 4326));
                    """
                    cur.execute(sql, (geojson_str,))
                elif bbox and len(bbox) == 4:
                    min_lon, min_lat, max_lon, max_lat = bbox
                    sql = """
                    SELECT tile_id FROM tiles
                    WHERE ST_Intersects(geometry, ST_MakeEnvelope(%s, %s, %s, %s, 4326));
                    """
                    cur.execute(sql, (min_lon, min_lat, max_lon, max_lat))
                else:
                    conn.close()
                    return None

                rows = cur.fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            log.warning(f"Spatial AOI pre-filter lookup encountered an issue: {e}. Skipping spatial pre-filter.")
            return None

    def build_qdrant_filter(
        self,
        filters: Optional[SearchFilter],
        candidate_tile_ids: Optional[List[str]] = None
    ) -> Optional[Filter]:
        """
        Builds a Qdrant Payload Filter from SearchFilter and spatial candidate IDs.
        """
        if not filters and not candidate_tile_ids:
            return None

        conditions = []

        # 1. Spatial Candidate Tile IDs (Phase 2.2 Task 3)
        if candidate_tile_ids is not None:
            if len(candidate_tile_ids) == 0:
                conditions.append(FieldCondition(key="tile_id", match=MatchValue(value="__NON_EXISTENT__")))
            elif len(candidate_tile_ids) == 1:
                conditions.append(FieldCondition(key="tile_id", match=MatchValue(value=candidate_tile_ids[0])))
            else:
                conditions.append(FieldCondition(key="tile_id", match=MatchAny(any=candidate_tile_ids)))

        if filters:
            # 2. Sensor Filter (Handle Sentinel-2 / Sentinel-2A / Sentinel-2B flexibly)
            if filters.sensor and filters.sensor.strip():
                s_val = filters.sensor.lower().strip()
                if "sentinel" in s_val:
                    conditions.append(
                        FieldCondition(
                            key="sensor",
                            match=MatchAny(any=[
                                "sentinel-2", "sentinel-2a", "sentinel-2b", "sentinel-2c", "sentinel-2d",
                                "Sentinel-2", "Sentinel-2A", "Sentinel-2B", "Sentinel-2C", "Sentinel-2D",
                                "Sentinel-2 L2A", "sentinel2", "sentinel", "Organiser Sensor"
                            ])
                        )
                    )
                elif "maxar" in s_val:
                    conditions.append(
                        FieldCondition(
                            key="sensor",
                            match=MatchAny(any=["Maxar", "maxar", "Maxar WorldView", "Maxar WorldView / Wayback"])
                        )
                    )
                else:
                    conditions.append(
                        FieldCondition(key="sensor", match=MatchValue(value=filters.sensor))
                    )

            # 3. Temporal Date-Range Filter (UNIX timestamp seconds)
            if filters.start_date or filters.end_date:
                date_range = {}
                if filters.start_date:
                    start_ts = int(filters.start_date.replace(tzinfo=timezone.utc).timestamp())
                    date_range["gte"] = start_ts
                if filters.end_date:
                    end_ts = int(filters.end_date.replace(tzinfo=timezone.utc).timestamp())
                    date_range["lte"] = end_ts
                conditions.append(
                    FieldCondition(key="acquisition_date", range=Range(**date_range))
                )

            # 4. Quality Gate Filter (min_quality)
            if filters.min_quality and filters.min_quality > 0.0:
                conditions.append(
                    FieldCondition(key="quality_confidence", range=Range(gte=filters.min_quality))
                )

            # 5. Cloud Percentage Filter (max_cloud_pct)
            if filters.max_cloud_pct is not None and filters.max_cloud_pct < 100.0:
                conditions.append(
                    FieldCondition(key="cloud_pct", range=Range(lte=filters.max_cloud_pct))
                )

        if not conditions:
            return None
        return Filter(must=conditions)

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[SearchFilter] = None,
        calibrate: bool = True,
        calibration_mode: str = "auto",
        return_details: bool = False
    ) -> Union[List[Tuple[str, float]], Tuple[List[Tuple[str, float]], Dict[str, Dict[str, Any]]]]:
        """
        Phase 2.2 Core kNN Search with QB-Norm Calibration:
        Executes filtered kNN search in Qdrant collections.
        Applies QB-Norm Dynamic Inverted Softmax using 512 RSICD Query Bank.
        Returns rank-ordered list of (tile_id, similarity_score).
        """
        candidate_tile_ids = None
        if filters and (filters.aoi_polygon or filters.bbox):
            candidate_tile_ids = self.resolve_aoi_to_tile_ids(
                aoi_polygon=filters.aoi_polygon,
                bbox=filters.bbox
            )
            if candidate_tile_ids is not None and len(candidate_tile_ids) == 0:
                log.info("AOI spatial filter resulted in 0 candidate tiles. Returning empty search results.")
                return ([], {}) if return_details else []

        q_filter = self.build_qdrant_filter(filters, candidate_tile_ids=candidate_tile_ids)

        # Dynamic Qdrant collection routing (Maxar vs Sentinel-2 vs Global)
        maxar_coll = os.getenv("QDRANT_MAXAR_COLLECTION", "maxar_tile_embeddings")
        collections_to_search = []
        if filters and filters.sensor:
            s_low = filters.sensor.lower()
            if "maxar" in s_low:
                collections_to_search = [maxar_coll]
            elif "sentinel" in s_low or "s2" in s_low:
                collections_to_search = [COLLECTION_NAME]
            else:
                collections_to_search = [COLLECTION_NAME, maxar_coll]
        else:
            collections_to_search = [COLLECTION_NAME, maxar_coll]

        fetch_limit = max(top_k * 8, 60)
        all_points = []
        for coll in collections_to_search:
            try:
                if hasattr(self.qdrant, "query_points"):
                    query_response = self.qdrant.query_points(
                        collection_name=coll,
                        query=query_vector,
                        query_filter=q_filter,
                        limit=fetch_limit,
                        with_payload=True,
                        with_vectors=True
                    )
                    all_points.extend(query_response.points)
                else:
                    pts = self.qdrant.search(
                        collection_name=coll,
                        query_vector=query_vector,
                        query_filter=q_filter,
                        limit=fetch_limit,
                        with_payload=True,
                        with_vectors=True
                    )
                    all_points.extend(pts)
            except Exception as e:
                log.warning(f"Vector search against '{coll}' encountered an issue: {e}")

        # Sort combined candidate points by similarity score descending
        all_points.sort(key=lambda p: float(p.score), reverse=True)

        candidate_ids: List[str] = []
        candidate_vecs: List[List[float]] = []
        raw_scores: List[float] = []
        seen_tile_ids = set()

        for p in all_points:
            tile_id = p.payload.get("tile_id") if p.payload else str(p.id)
            if not tile_id or tile_id in seen_tile_ids:
                continue
            seen_tile_ids.add(tile_id)
            score = float(p.score)

            vec = p.vector
            if isinstance(vec, dict):
                vec = vec.get("image", None) or (list(vec.values())[0] if vec else None)

            candidate_ids.append(tile_id)
            raw_scores.append(score)
            if vec is not None:
                candidate_vecs.append(vec)

        if not candidate_ids:
            return ([], {}) if return_details else []

        # QB-Norm Dynamic Inverted Softmax Calibration (Bogolin et al. CVPR 2022)
        calibrator = get_calibrator()
        can_calibrate = (
            calibrate
            and calibrator.is_ready
            and len(candidate_vecs) == len(candidate_ids)
            and len(candidate_ids) > 0
        )

        calibration_meta: Dict[str, Dict[str, Any]] = {}

        if can_calibrate:
            q_arr = np.array(query_vector, dtype=np.float32)
            c_vecs_arr = np.array(candidate_vecs, dtype=np.float32)
            r_scores_arr = np.array(raw_scores, dtype=np.float32)

            ranked_ids, calib_scores, raw_sorted, hub_flags = calibrator.calibrate_candidates(
                query_vec=q_arr,
                candidate_ids=candidate_ids,
                candidate_vecs=c_vecs_arr,
                raw_scores=r_scores_arr,
                mode=calibration_mode
            )

            results = list(zip(ranked_ids, [float(s) for s in calib_scores]))
            calibration_meta = {
                tid: {
                    "raw_score": float(r),
                    "calibrated_score": float(c),
                    "is_hub": bool(h),
                    "calibrated": True
                }
                for tid, c, r, h in zip(ranked_ids, calib_scores, raw_sorted, hub_flags)
            }
        else:
            results = list(zip(candidate_ids, raw_scores))
            calibration_meta = {
                tid: {
                    "raw_score": float(r),
                    "calibrated_score": float(r),
                    "is_hub": False,
                    "calibrated": False
                }
                for tid, r in zip(candidate_ids, raw_scores)
            }

        if return_details:
            return results, calibration_meta
        return results

    def hydrate_from_postgres(
        self,
        ranked_tile_pairs: List[Tuple[str, float]],
        top_k: int = 5,
        deduplicate_spatial: bool = True,
        calibration_meta: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> List[SearchResultItem]:
        """
        Phase 2.3: Batch-fetches tile metadata and geometry in a single SQL query
        and preserves Qdrant's similarity score ranking order.
        Phase 2.4: Attaches auto-generated 3-index spot description to each item.
        Performs spatial deduplication so overlapping granules and repeat timestamps of the
        same physical spot (site_key) are merged, returning top_k distinct physical locations.
        """
        if not ranked_tile_pairs:
            return []

        tile_scores = {t_id: score for t_id, score in ranked_tile_pairs}
        tile_ids_list = [t_id for t_id, _ in ranked_tile_pairs]

        try:
            conn = self._get_pg_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                sql = """
                SELECT
                    t.tile_id,
                    t.scene_id,
                    t.site_key,
                    t.acquisition_date,
                    t.sensor,
                    t.cloud_pct,
                    t.quality_confidence,
                    t.mean_ndvi,
                    t.mean_ndwi,
                    t.mean_ndbi,
                    t.centroid_lat,
                    t.centroid_lon,
                    t.file_path,
                    t.thumbnail_path,
                    ST_AsGeoJSON(t.geometry)::json AS geometry_geojson
                FROM tiles t
                WHERE t.tile_id IN %s;
                """
                cur.execute(sql, (tuple(tile_ids_list),))
                rows = cur.fetchall()
            conn.close()

            # Map by tile_id for fast lookup
            rows_by_id = {r["tile_id"]: r for r in rows}

            def _parse_acq_ts(row_data: Dict[str, Any]) -> float:
                ad = row_data.get("acquisition_date")
                if isinstance(ad, datetime):
                    return ad.timestamp()
                if isinstance(ad, str):
                    try:
                        return datetime.fromisoformat(ad.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        pass
                return 0.0

            if deduplicate_spatial:
                # Group all candidate matches by physical ground spot (site_key or ~200m spatial proximity)
                spot_groups: Dict[str, List[Tuple[str, float, Dict[str, Any], float]]] = {}
                spot_centroids: List[Tuple[float, float, str]] = []  # (lat, lon, spot_key)
                next_cluster_idx = 0

                for t_id, score in ranked_tile_pairs:
                    r = rows_by_id.get(t_id)
                    if not r:
                        continue

                    site_key = r.get("site_key")
                    c_lat = r.get("centroid_lat")
                    c_lon = r.get("centroid_lon")
                    ts = _parse_acq_ts(r)

                    assigned_spot = None
                    if site_key and site_key not in ("null", "None"):
                        assigned_spot = f"site:{site_key}"
                    elif c_lat is not None and c_lon is not None:
                        for prev_lat, prev_lon, prev_spot in spot_centroids:
                            if abs(c_lat - prev_lat) < 0.002 and abs(c_lon - prev_lon) < 0.002:
                                assigned_spot = prev_spot
                                break

                    if assigned_spot is None:
                        assigned_spot = f"geo_cluster:{next_cluster_idx}"
                        next_cluster_idx += 1
                        if c_lat is not None and c_lon is not None:
                            spot_centroids.append((c_lat, c_lon, assigned_spot))

                    if assigned_spot not in spot_groups:
                        spot_groups[assigned_spot] = []
                    spot_groups[assigned_spot].append((t_id, score, r, ts))

                # For each physical spot, select the LATEST observation (highest timestamp).
                # If timestamps are identical, break tie with highest similarity score.
                selected_spot_candidates: List[Tuple[str, float, Dict[str, Any]]] = []
                for spot_id, group in spot_groups.items():
                    group.sort(key=lambda x: (x[3], x[1]), reverse=True)
                    best_candidate = group[0]
                    # Retain the maximum similarity score across observations of this spot for ranking
                    max_sim = max(x[1] for x in group)
                    selected_spot_candidates.append((best_candidate[0], max_sim, best_candidate[2]))

                # Rank the distinct physical spots by their similarity score descending
                selected_spot_candidates.sort(key=lambda x: x[1], reverse=True)
                candidates_to_process = selected_spot_candidates[:top_k]
            else:
                candidates_to_process = [
                    (t_id, score, rows_by_id[t_id])
                    for t_id, score in ranked_tile_pairs
                    if t_id in rows_by_id
                ][:top_k]

            items = []
            for t_id, score, r in candidates_to_process:
                acq_str = r["acquisition_date"].isoformat() if isinstance(r["acquisition_date"], datetime) else str(r["acquisition_date"])
                thumb_path = r.get("thumbnail_path")
                file_path = r.get("file_path")
                site_key = r.get("site_key")

                # Robust thumbnail URL resolution
                thumb_url = None
                if thumb_path:
                    clean_tp = thumb_path.replace("\\", "/")
                    if "/data/" in clean_tp:
                        thumb_url = "/data/" + clean_tp.split("/data/", 1)[1]
                    elif clean_tp.startswith("data/"):
                        thumb_url = "/" + clean_tp
                    elif clean_tp.startswith("/"):
                        thumb_url = clean_tp
                    else:
                        thumb_url = f"/data/{clean_tp}"
                elif file_path:
                    clean_fp = file_path.replace("\\", "/")
                    if clean_fp.endswith(".tif"):
                        cand = clean_fp[:-4] + "_preview.jpg"
                        if "/data/" in cand:
                            thumb_url = "/data/" + cand.split("/data/", 1)[1]
                        else:
                            thumb_url = f"/data/{cand}"
                elif site_key and site_key not in ("null", "None"):
                    thumb_url = f"/data/tiles/{site_key}/{t_id}_preview.jpg"
                else:
                    thumb_url = f"/data/tiles/{t_id}_preview.jpg"

                # Generate Phase 2.4 plain-language spot description
                description = generate_tile_description(
                    mean_ndvi=r.get("mean_ndvi"),
                    mean_ndwi=r.get("mean_ndwi"),
                    mean_ndbi=r.get("mean_ndbi"),
                    sensor=r.get("sensor")
                )

                meta = calibration_meta.get(t_id, {}) if calibration_meta else {}
                raw_score = meta.get("raw_score", score)
                is_hub = meta.get("is_hub", False)
                is_calibrated = meta.get("calibrated", False)

                items.append(
                    SearchResultItem(
                        tile_id=t_id,
                        score=round(score, 4),
                        raw_score=round(raw_score, 4),
                        is_hub=is_hub,
                        calibrated=is_calibrated,
                        scene_id=r["scene_id"],
                        site_key=r["site_key"],
                        acquisition_date=acq_str,
                        centroid_lat=r["centroid_lat"],
                        centroid_lon=r["centroid_lon"],
                        sensor=r["sensor"],
                        cloud_pct=r["cloud_pct"],
                        quality_confidence=r["quality_confidence"],
                        mean_ndvi=r["mean_ndvi"],
                        mean_ndwi=r["mean_ndwi"],
                        mean_ndbi=r["mean_ndbi"],
                        spot_description=description,
                        file_path=r["file_path"],
                        thumbnail_url=thumb_url,
                        thumbnail_path=thumb_path,
                        geometry_geojson=r["geometry_geojson"]
                    )
                )

            return items

        except Exception as e:
            log.error(f"Failed to hydrate search results from PostgreSQL: {e}", exc_info=True)
            fallback_items = []
            for t_id, score in ranked_tile_pairs[:top_k]:
                meta = calibration_meta.get(t_id, {}) if calibration_meta else {}
                fallback_items.append(
                    SearchResultItem(
                        tile_id=t_id,
                        score=round(score, 4),
                        raw_score=round(meta.get("raw_score", score), 4),
                        is_hub=meta.get("is_hub", False),
                        calibrated=meta.get("calibrated", False)
                    )
                )
            return fallback_items

    def log_search(
        self,
        analyst_id: str,
        raw_query: str,
        query_type: str,
        filters: Optional[SearchFilter],
        result_tile_ids: List[str],
        query_embedding: Optional[List[float]] = None
    ):
        """
        Phase 2.5: Logs search history into Postgres `search_log` table
        for future feedback-driven reranking and audit lineage.
        Preserves the exact 512-D query_embedding vector for lossless Rocchio steering.
        """
        search_log_id = None
        try:
            conn = self._get_pg_conn()
            filters_json = "{}"
            if filters:
                if hasattr(filters, "model_dump"):
                    filters_json = json.dumps(filters.model_dump(exclude_none=True), default=str)
                elif hasattr(filters, "dict"):
                    filters_json = json.dumps(filters.dict(exclude_none=True), default=str)
                elif isinstance(filters, dict):
                    filters_json = json.dumps(filters, default=str)
            with conn.cursor() as cur:
                sql = """
                INSERT INTO search_log (analyst_id, raw_query, query_type, filters, result_tile_ids, query_embedding)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING search_id;
                """
                cur.execute(sql, (analyst_id, raw_query, query_type, filters_json, result_tile_ids, query_embedding))
                row = cur.fetchone()
                if row:
                    search_log_id = str(row[0])
            conn.commit()
            conn.close()
        except Exception as e:
            log.warning(f"Could not log search to search_log table: {e}")
        return search_log_id

    def search(self, request: SearchRequest) -> SearchResponse:
        """
        Phase 2.5: Unified SemanticSearchTool entrypoint.
        Executes query encoding, filtered vector kNN search, Postgres hydration,
        3-index spot description generation, and search audit logging.
        """
        t0 = time.time()

        # Step 1: Query Encoding (Phase 2.1)
        if request.query_text:
            query_type = "text"
            query_repr = request.query_text
            query_vec = encode_query_text(request.query_text)
        elif request.query_image_path:
            query_type = "image"
            query_repr = f"file:{os.path.basename(request.query_image_path)}"
            query_vec = encode_query_image(request.query_image_path)
        elif request.query_image_bytes:
            query_type = "image"
            import re
            clean_fn = request.query_filename or f"query_{int(time.time())}.png"
            clean_base = re.sub(r'[^a-zA-Z0-9_\-\.]', '_', os.path.basename(clean_fn))
            data_root = os.getenv("DATA_DIR", "data")
            save_dir = os.path.join(data_root, "uploads")
            os.makedirs(save_dir, exist_ok=True)
            stored_name = f"{int(time.time())}_{clean_base}"
            save_path = os.path.join(save_dir, stored_name)
            try:
                with open(save_path, "wb") as f:
                    f.write(request.query_image_bytes)
                query_repr = f"image:{stored_name}"
            except Exception as e:
                log.warning(f"Could not persist uploaded query image: {e}")
                query_repr = f"image:{clean_base}"
            query_vec = encode_query_image(request.query_image_bytes)
        else:
            raise ValueError("Invalid SearchRequest: Must provide query_text, query_image_path, or query_image_bytes.")

        # Step 1.5: Automatic Rocchio Relevance Feedback Steering & Penalization
        feedback_tiles = {}
        try:
            from backend.services.analyst_workflow import get_search_query_feedback
            feedbacks = get_search_query_feedback(query_repr)
            if not feedbacks and query_type == "image":
                feedbacks = get_search_query_feedback("raw_image_bytes")
            if feedbacks:
                for fb in feedbacks:
                    feedback_tiles[fb["tile_id"]] = fb["relevant"]

                pos_tile_ids = [tid for tid, rel in feedback_tiles.items() if rel]
                neg_tile_ids = [tid for tid, rel in feedback_tiles.items() if not rel]

                if pos_tile_ids or neg_tile_ids:
                    from qdrant_client.models import Filter, FieldCondition, MatchValue
                    maxar_coll = os.getenv("QDRANT_MAXAR_COLLECTION", "maxar_tile_embeddings")
                    pos_vecs = []
                    neg_vecs = []

                    def _get_vec(tid):
                        for c in [COLLECTION_NAME, maxar_coll]:
                            try:
                                flt = Filter(must=[FieldCondition(key="tile_id", match=MatchValue(value=tid))])
                                pts = self.qdrant.scroll(collection_name=c, scroll_filter=flt, with_vectors=True, limit=1)[0]
                                if pts and pts[0].vector is not None:
                                    return np.array(pts[0].vector, dtype=np.float32)
                            except Exception:
                                pass
                        return None

                    for tid in pos_tile_ids:
                        v = _get_vec(tid)
                        if v is not None:
                            pos_vecs.append(v)
                    for tid in neg_tile_ids:
                        v = _get_vec(tid)
                        if v is not None:
                            neg_vecs.append(v)

                    # Non-destructive Rocchio Feedback:
                    # Steer towards confirmed targets without inverting the query on negatives
                    q_mod = np.array(query_vec, dtype=np.float32)
                    if pos_vecs:
                        q_mod += 0.50 * np.mean(pos_vecs, axis=0)
                        if neg_vecs:
                            q_mod -= 0.15 * np.mean(neg_vecs, axis=0)
                    elif neg_vecs:
                        # Mild directional nudge away from noise cluster; preserves query manifold
                        q_mod -= 0.08 * np.mean(neg_vecs, axis=0)

                    norm = np.linalg.norm(q_mod)
                    if norm > 1e-6:
                        query_vec = (q_mod / norm).tolist()
                        log.info(f"[Rocchio] Non-destructive steering for '{query_repr}': +{len(pos_vecs)} pos, -{len(neg_vecs)} neg")
        except Exception as e:
            log.warning(f"Rocchio feedback steering skipped: {e}")

        # Step 2: Vector Search with RSICD 512 Query Bank Calibration
        # Fetch generous candidate pool so analyst-rejected tiles are replaced with new matching candidates
        neg_count = len([rel for rel in feedback_tiles.values() if not rel])
        extra_k = max(neg_count * 3, 20) + (5 if query_type == "image" else 0)
        fetch_k = max(request.top_k + extra_k, 35)
        ranked_pairs, calib_meta = self.search_vectors(
            query_vector=query_vec,
            top_k=fetch_k,
            filters=request.filters,
            calibrate=request.calibrate,
            calibration_mode=request.calibration_mode,
            return_details=True
        )

        # Step 3: Hydrate & Describe Results (Phases 2.3 & 2.4)
        results = self.hydrate_from_postgres(
            ranked_tile_pairs=ranked_pairs,
            top_k=fetch_k,
            deduplicate_spatial=True,
            calibration_meta=calib_meta
        )

        # Self-match exclusion for image-to-image queries:
        # When an analyst uploads an image to find similar scenes, exclude the query image itself!
        if query_type == "image":
            clean_q_fn = request.query_filename or (
                os.path.basename(request.query_image_path) if request.query_image_path else ""
            )
            clean_stem = ""
            if clean_q_fn:
                import re
                clean_stem = re.sub(r'(_thumb|_preview)?\.(jpg|jpeg|png|tif|tiff)$', '', clean_q_fn.lower())

            filtered_results = []
            excluded_self = False

            for r in results:
                is_self = False
                # 1. Similarity score >= 0.988 indicates identical image vector in 512-dim RemoteCLIP space
                if r.score >= 0.988:
                    is_self = True
                # 2. Uploaded filename match with tile_id or file paths
                elif clean_stem:
                    r_tid_low = r.tile_id.lower()
                    if clean_stem == r_tid_low or clean_stem in r_tid_low or r_tid_low in clean_stem:
                        is_self = True
                    elif r.thumbnail_path and clean_stem in r.thumbnail_path.lower():
                        is_self = True
                    elif r.file_path and clean_stem in r.file_path.lower():
                        is_self = True

                if is_self and not excluded_self:
                    log.info(f"[SemanticRetrieval] Excluded uploaded query image itself from results: tile_id='{r.tile_id}', score={r.score}")
                    excluded_self = True
                    continue

                filtered_results.append(r)

            results = filtered_results

        # Algorithmic Feedback filtering:
        # Target only the rejected tiles: drop them so fresh matching candidates from the vector space take their slots!
        if feedback_tiles:
            neg_ids = {tid for tid, rel in feedback_tiles.items() if not rel}
            if neg_ids:
                results = [r for r in results if r.tile_id not in neg_ids]

        default_cutoff = 0.60 if query_type == "image" else 0.15
        cutoff = request.min_similarity if request.min_similarity is not None else default_cutoff
        passed_results = [r for r in results if r.score >= cutoff]
        if not passed_results and results:
            results = results[:request.top_k]
        else:
            results = passed_results[:request.top_k]

        # Step 4: Audit Logging (Phase 2.5)
        result_ids = [r.tile_id for r in results]
        query_emb_list = None
        try:
            if isinstance(query_vec, np.ndarray):
                query_emb_list = [float(x) for x in query_vec.flatten()]
            elif isinstance(query_vec, list):
                query_emb_list = [float(x) for x in query_vec]
        except Exception as e:
            log.warning(f"Could not convert query_vec for storage: {e}")

        logged_id = self.log_search(
            analyst_id=request.analyst_id,
            raw_query=query_repr,
            query_type=query_type,
            filters=request.filters,
            result_tile_ids=result_ids,
            query_embedding=query_emb_list
        )

        elapsed_ms = (time.time() - t0) * 1000.0

        return SearchResponse(
            search_log_id=logged_id,
            query_type=query_type,
            query=query_repr,
            total_found=len(results),
            filters_applied=request.filters.dict(exclude_none=True) if request.filters else {},
            results=results,
            execution_time_ms=round(elapsed_ms, 2)
        )


# ============================================================
# Global Singleton & Helper Method
# ============================================================

_global_vector_search_service: Optional[VectorSearchService] = None


def get_vector_search_service() -> VectorSearchService:
    """
    Obtain shared singleton instance of VectorSearchService.
    """
    global _global_vector_search_service
    if _global_vector_search_service is None:
        _global_vector_search_service = VectorSearchService()
    return _global_vector_search_service


def search_semantic(
    query: Union[str, bytes, Path, np.ndarray],
    top_k: int = 5,
    filters: Optional[SearchFilter] = None,
    analyst_id: str = "demo_analyst"
) -> SearchResponse:
    """
    Convenience function for agent or REST endpoints to run semantic search.
    """
    service = get_vector_search_service()

    if isinstance(query, str) and not os.path.exists(query):
        req = SearchRequest(query_text=query, top_k=top_k, filters=filters, analyst_id=analyst_id)
    elif isinstance(query, str) and os.path.exists(query):
        req = SearchRequest(query_image_path=query, top_k=top_k, filters=filters, analyst_id=analyst_id)
    elif isinstance(query, bytes):
        req = SearchRequest(query_image_bytes=query, top_k=top_k, filters=filters, analyst_id=analyst_id)
    else:
        req = SearchRequest(query_text=str(query), top_k=top_k, filters=filters, analyst_id=analyst_id)

    return service.search(req)
