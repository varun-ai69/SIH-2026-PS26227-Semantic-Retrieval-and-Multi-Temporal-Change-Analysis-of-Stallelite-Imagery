"""
backend/api/routers/discovery.py
================================
Phase 3: FastAPI Router for Discovery & Clustering (PS 2.2.4)
=============================================================

Endpoints:
  - GET /api/v1/discover/{tile_id}
      Discovers visually and semantically similar tiles across the entire archive.
  - GET /api/v1/clusters
      Returns summary of all discovered clusters with representative medoids.
  - GET /api/v1/clusters/{cluster_id}/tiles
      Returns member tiles belonging to a specific cluster.
  - POST /api/v1/clusters/recompute
      Triggers the standalone clustering job across the entire archive.
"""

import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.services.discovery import get_discovery_service
from backend.services.vector_search import SearchResultItem
from backend.jobs.run_clustering import run_clustering_job

log = logging.getLogger("DiscoveryRouter")

router = APIRouter(prefix="/api/v1", tags=["Discovery & Clustering"])


@router.get(
    "/discover/{tile_id}",
    response_model=List[SearchResultItem],
    summary="Generic Similarity Discovery (PS 2.2.4)"
)
def discover_similar_tiles(
    tile_id: str,
    top_k: int = Query(10, ge=1, le=50, description="Maximum number of similar tiles to discover"),
    min_similarity: float = Query(0.40, ge=0.0, le=1.0, description="Minimum cosine similarity cutoff")
):
    """
    Given any existing tile_id from anywhere in the system, uses its precomputed
    stored vector to discover visual & semantic twins across all ingested regions.
    Zero prompt, zero new embedding computation.
    """
    try:
        service = get_discovery_service()
        results = service.find_similar_tiles(
            tile_id=tile_id,
            top_k=top_k,
            min_similarity=min_similarity
        )
        return results
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        log.error(f"Discovery failed for tile '{tile_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Discovery failed: {str(e)}")


@router.get(
    "/clusters",
    response_model=List[Dict[str, Any]],
    summary="List Discovered Terrain & Land-Use Clusters"
)
def list_clusters():
    """
    Returns all clusters discovered across the archive with representative tile thumbnails,
    spectral signatures, and tile counts.
    """
    try:
        service = get_discovery_service()
        return service.get_clusters_summary()
    except Exception as e:
        log.error(f"Failed to fetch clusters: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch clusters: {str(e)}")


@router.get(
    "/clusters/{cluster_id}/tiles",
    response_model=List[SearchResultItem],
    summary="Get Member Tiles of a Cluster"
)
def get_cluster_tiles(
    cluster_id: str,
    limit: int = Query(60, ge=1, le=120, description="Max tiles to return for cluster")
):
    """
    Returns all member tiles belonging to a specific cluster.
    """
    try:
        service = get_discovery_service()
        return service.get_cluster_tiles(cluster_id=cluster_id, limit=limit)
    except Exception as e:
        log.error(f"Failed to fetch tiles for cluster '{cluster_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to fetch cluster tiles: {str(e)}")


@router.post(
    "/clusters/recompute",
    response_model=Dict[str, Any],
    summary="Trigger Offline Re-Clustering Job"
)
def recompute_clusters(
    n_clusters: int = Query(8, ge=2, le=32, description="Target number of semantic clusters (default 8)"),
    method: str = Query("kmeans", description="Clustering algorithm: 'kmeans' or 'hdbscan'")
):
    """
    Triggers the batch KMeans/HDBSCAN clustering job over all vectors in Qdrant.
    Refreshes cluster IDs in Postgres 'tiles' and 'clusters' tables and updates Qdrant payloads.
    Default setting creates 8 coherent semantic clusters with vision-language labels.
    """
    try:
        log.info(f"Triggering clustering job via API (n_clusters={n_clusters}, method={method})...")
        result = run_clustering_job(target_clusters=n_clusters, method=method)
        return result
    except Exception as e:
        log.error(f"Clustering job execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Clustering job failed: {str(e)}")
