"""
backend/services/discovery.py
==============================
Phase 2 & 3: Generic Similarity Discovery Service (PS 2.2.4)
===========================================================

Provides the core discovery functions:
1. find_similar_tiles(tile_id, top_k, min_similarity):
   Input: Any existing tile_id from anywhere in the system.
   Action: Zero new prompt, zero re-encoding. Pulls the tile's stored 512-dim vector
   directly from Qdrant and finds the closest visual/semantic twins across the entire archive.
   Output: List[SearchResultItem] enriched with Postgres metadata and spectral explanations.
2. get_clusters_summary():
   Returns all clusters from Postgres 'clusters' table with medoid tile thumbnails & stats.
3. get_cluster_tiles(cluster_id, limit):
   Returns member tiles of a given cluster.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient

from backend.services.vector_store import (
    tile_id_to_uuid,
    get_qdrant_client,
    COLLECTION_NAME,
    MAXAR_COLLECTION_NAME
)
from backend.services.vector_search import (
    SearchResultItem,
    get_vector_search_service,
    PG_DSN
)

log = logging.getLogger("DiscoveryService")


def resolve_thumbnail_url(raw_path: Optional[str]) -> Optional[str]:
    """
    Normalizes local disk or container paths to web-accessible /data/ URLs.
    Prevents double-slash // protocol-relative URL bugs that cause ERR_NAME_NOT_RESOLVED.
    """
    if not raw_path:
        return None
    clean = raw_path.replace("\\", "/").strip()
    if "/data/" in clean:
        return "/data/" + clean.split("/data/", 1)[1]
    if clean.startswith("data/"):
        return "/" + clean
    if clean.startswith("/app/data/"):
        return "/data/" + clean[10:]
    if clean.startswith("app/data/"):
        return "/data/" + clean[9:]
    if clean.startswith("/"):
        return clean
    return f"/data/{clean}"


class DiscoveryService:
    """
    Generic Similarity Discovery Service.
    Decoupled from spatial restrictions — searches across all ingested regions globally.
    """

    def __init__(self, qdrant_client: Optional[QdrantClient] = None):
        self.qdrant = qdrant_client or get_qdrant_client()
        self.vector_search = get_vector_search_service()

    def _get_pg_conn(self):
        return psycopg2.connect(PG_DSN)

    def get_tile_vector(self, tile_id: str) -> Optional[List[float]]:
        """
        Retrieves the 512-dim embedding for a tile from Qdrant without re-encoding.
        Checks both Sentinel-2 and Maxar collections.
        """
        point_uuid = tile_id_to_uuid(tile_id)
        target_collections = [COLLECTION_NAME, MAXAR_COLLECTION_NAME]

        for coll in target_collections:
            try:
                pts = self.qdrant.retrieve(
                    collection_name=coll,
                    ids=[point_uuid],
                    with_vectors=True,
                    with_payload=True
                )
                if pts and len(pts) > 0 and pts[0].vector is not None:
                    return list(pts[0].vector)
            except Exception as e:
                log.debug(f"Point lookup in {coll} for {tile_id} failed: {e}")

        # Fallback: query by payload filter if UUID did not match
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        for coll in target_collections:
            try:
                res = self.qdrant.scroll(
                    collection_name=coll,
                    scroll_filter=Filter(
                        must=[FieldCondition(key="tile_id", match=MatchValue(value=tile_id))]
                    ),
                    limit=1,
                    with_vectors=True
                )
                points, _ = res
                if points and len(points) > 0 and points[0].vector is not None:
                    return list(points[0].vector)
            except Exception as e:
                log.debug(f"Payload scroll in {coll} for {tile_id} failed: {e}")

        return None

    def find_similar_tiles(
        self,
        tile_id: str,
        top_k: int = 10,
        min_similarity: float = 0.50
    ) -> List[SearchResultItem]:
        """
        Core PS 2.2.4 Generic Discovery Function.
        - Input: Any existing tile_id in the system.
        - Output: Ranked list of visually and semantically similar tiles across the entire archive.
        - Zero new encoding computation.
        """
        log.info(f"Executing similarity discovery for seed tile: '{tile_id}' (top_k={top_k})...")

        # 1. Pull seed vector
        seed_vector = self.get_tile_vector(tile_id)
        if not seed_vector:
            raise ValueError(f"Tile '{tile_id}' vector not found in Qdrant archive.")

        # 2. Search across collections for nearest neighbors
        # Fetch top_k + 5 to account for excluding the seed tile itself
        fetch_limit = max(top_k + 5, 20)
        target_collections = [COLLECTION_NAME, MAXAR_COLLECTION_NAME]

        all_points = []
        for coll in target_collections:
            try:
                if hasattr(self.qdrant, "query_points"):
                    query_response = self.qdrant.query_points(
                        collection_name=coll,
                        query=seed_vector,
                        limit=fetch_limit,
                        with_payload=True
                    )
                    all_points.extend(query_response.points)
                else:
                    pts = self.qdrant.search(
                        collection_name=coll,
                        query_vector=seed_vector,
                        limit=fetch_limit,
                        with_payload=True
                    )
                    all_points.extend(pts)
            except Exception as e:
                log.warning(f"Discovery vector search in '{coll}' encountered an issue: {e}")

        # 3. Sort candidates by similarity descending & filter out seed tile
        all_points.sort(key=lambda p: float(p.score), reverse=True)

        ranked_pairs: List[Tuple[str, float]] = []
        seen_tile_ids = {tile_id}  # Exclude seed tile itself

        for p in all_points:
            t_id = p.payload.get("tile_id") if p.payload else str(p.id)
            score = float(p.score)

            if score < min_similarity:
                continue

            if t_id and t_id not in seen_tile_ids:
                seen_tile_ids.add(t_id)
                ranked_pairs.append((t_id, score))

        if not ranked_pairs:
            log.info(f"No similar tiles found for '{tile_id}' above similarity cutoff {min_similarity}.")
            return []

        # 4. Hydrate metadata from Postgres using existing tested vector_search service
        hydrated_results = self.vector_search.hydrate_from_postgres(
            ranked_tile_pairs=ranked_pairs[:top_k * 2],
            top_k=top_k,
            deduplicate_spatial=True
        )

        log.info(f"Discovery found {len(hydrated_results)} similar tiles for seed '{tile_id}'.")
        return hydrated_results

    def get_clusters_summary(self) -> List[Dict[str, Any]]:
        """
        Queries Postgres for all discovered clusters, joining representative tile
        metadata to render rich cluster cards in the Workbench.
        """
        try:
            conn = self._get_pg_conn()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                sql = """
                SELECT
                    c.cluster_id,
                    c.label,
                    c.representative_tile_id,
                    c.tile_count,
                    c.computed_at,
                    c.model_version,
                    t.thumbnail_path,
                    t.centroid_lat,
                    t.centroid_lon,
                    t.mean_ndvi,
                    t.mean_ndwi,
                    t.mean_ndbi,
                    t.sensor,
                    t.scene_id
                FROM clusters c
                LEFT JOIN tiles t ON c.representative_tile_id = t.tile_id
                ORDER BY c.tile_count DESC;
                """
                cur.execute(sql)
                rows = cur.fetchall()
            conn.close()

            results = []
            for r in rows:
                results.append({
                    "cluster_id": r["cluster_id"],
                    "label": r["label"] or f"Cluster {r['cluster_id']}",
                    "representative_tile_id": r["representative_tile_id"],
                    "tile_count": r["tile_count"] or 0,
                    "computed_at": r["computed_at"].isoformat() if r["computed_at"] else None,
                    "model_version": r["model_version"],
                    "thumbnail_url": resolve_thumbnail_url(r.get("thumbnail_path")),
                    "centroid_lat": r.get("centroid_lat"),
                    "centroid_lon": r.get("centroid_lon"),
                    "mean_ndvi": r.get("mean_ndvi"),
                    "mean_ndwi": r.get("mean_ndwi"),
                    "mean_ndbi": r.get("mean_ndbi"),
                    "sensor": r.get("sensor"),
                    "scene_id": r.get("scene_id")
                })
            return results
        except Exception as e:
            log.error(f"Failed to fetch clusters summary: {e}", exc_info=True)
            return []

    def get_cluster_tiles(self, cluster_id: str, limit: int = 60) -> List[SearchResultItem]:
        """
        Retrieves member tiles for a given cluster_id formatted as SearchResultItem.
        """
        try:
            conn = self._get_pg_conn()
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT tile_id FROM tiles WHERE cluster_id = %s ORDER BY acquisition_date DESC, tile_id ASC LIMIT %s;",
                    (cluster_id, limit)
                )
                rows = cur.fetchall()
            conn.close()

            tile_ids = [r[0] for r in rows]
            if not tile_ids:
                return []

            # Dummy score 1.0 since they belong to the same cluster
            ranked_pairs = [(t_id, 1.0) for t_id in tile_ids]
            return self.vector_search.hydrate_from_postgres(
                ranked_tile_pairs=ranked_pairs,
                top_k=limit,
                deduplicate_spatial=False
            )
        except Exception as e:
            log.error(f"Failed to fetch tiles for cluster '{cluster_id}': {e}", exc_info=True)
            return []


# Singleton accessor
_DISCOVERY_SERVICE: Optional[DiscoveryService] = None

def get_discovery_service() -> DiscoveryService:
    global _DISCOVERY_SERVICE
    if _DISCOVERY_SERVICE is None:
        _DISCOVERY_SERVICE = DiscoveryService()
    return _DISCOVERY_SERVICE
