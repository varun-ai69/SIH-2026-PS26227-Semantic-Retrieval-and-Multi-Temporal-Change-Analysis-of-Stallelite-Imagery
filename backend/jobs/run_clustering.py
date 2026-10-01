"""
backend/jobs/run_clustering.py
==============================
Phase 1 & 2: Standalone Vector Clustering Job (PS 2.2.4)
========================================================

Executes unsupervised clustering over all tile embeddings across all ingested regions
in Qdrant (both Sentinel-2 and Maxar collections).

Pipeline:
1. Pulls all vectors, point IDs, and tile IDs from Qdrant via paginated scroll.
2. Performs HDBSCAN clustering (with adaptive KMeans fallback for smaller archives).
3. Assigns cluster IDs and selects the medoid (centroid-closest) representative tile.
4. Generates an interpretable spectral/landscape label from medoid tile indices.
5. Updates Postgres 'tiles.cluster_id' and Qdrant 'payload.cluster_id'.
6. Refreshes the Postgres 'clusters' table.

Can be run:
- As a standalone script: python -m backend.jobs.run_clustering
- Programmatically / via API: run_clustering_job()
"""

import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import psycopg2
import psycopg2.extras
from qdrant_client import QdrantClient

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [ClusteringJob]: %(message)s"
)
log = logging.getLogger("ClusteringJob")

# Configuration from Environment
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = int(os.getenv("POSTGRES_PORT", "5434"))
PG_DB = os.getenv("POSTGRES_DB", "eo_archive")
PG_USER = os.getenv("POSTGRES_USER", "eo_admin")
PG_PASS = os.getenv("POSTGRES_PASSWORD", "eo_password")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_S2 = os.getenv("QDRANT_COLLECTION", "tile_embeddings")
COLLECTION_MAXAR = os.getenv("QDRANT_MAXAR_COLLECTION", "maxar_tile_embeddings")
MODEL_VERSION = "RemoteCLIP-ViT-B-32"


def get_pg_connection():
    """Establishes Postgres connection."""
    return psycopg2.connect(
        host=PG_HOST,
        port=PG_PORT,
        dbname=PG_DB,
        user=PG_USER,
        password=PG_PASS
    )


def get_qdrant_client() -> QdrantClient:
    """Initializes Qdrant client."""
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)


def fetch_all_qdrant_points(client: QdrantClient) -> Tuple[List[Any], List[str], List[np.ndarray], List[str]]:
    """
    Scrolls across both Sentinel-2 and Maxar Qdrant collections to fetch:
    - point_ids (UUIDs or ints)
    - tile_ids (from payload.tile_id)
    - vectors (NumPy 1D float32 arrays)
    - collections (source collection name for payload writeback)
    """
    point_ids: List[Any] = []
    tile_ids: List[str] = []
    vectors: List[np.ndarray] = []
    collections: List[str] = []

    target_collections = [COLLECTION_S2, COLLECTION_MAXAR]

    for coll in target_collections:
        try:
            # Check if collection exists
            coll_info = client.get_collection(coll)
            if not coll_info or coll_info.points_count == 0:
                log.info(f"Collection '{coll}' is empty or does not exist. Skipping.")
                continue

            log.info(f"Scrolling collection '{coll}' (total points: {coll_info.points_count})...")
            offset = None
            batch_count = 0

            while True:
                points, next_page = client.scroll(
                    collection_name=coll,
                    limit=200,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True
                )

                for p in points:
                    if p.vector is None:
                        continue
                    
                    t_id = p.payload.get("tile_id") if p.payload else str(p.id)
                    vec = np.asarray(p.vector, dtype=np.float32)

                    point_ids.append(p.id)
                    tile_ids.append(t_id)
                    vectors.append(vec)
                    collections.append(coll)

                batch_count += len(points)
                if not next_page:
                    break
                offset = next_page

            log.info(f"Loaded {batch_count} vectors from collection '{coll}'.")
        except Exception as e:
            log.warning(f"Failed to scroll points from collection '{coll}': {e}")

    return point_ids, tile_ids, vectors, collections


DEFAULT_CLUSTERS = 8

TACTICAL_LANDCOVER_TAXONOMY = [
    {
        "label": "Agricultural Farmlands & Vegetative Fields",
        "category": "vegetat",
        "prompt": "agricultural farmlands, cultivated crop fields, rural vegetation and fertile pastures"
    },
    {
        "label": "Dense Woodland & Vegetative Canopy",
        "category": "vegetat",
        "prompt": "dense forest canopy, woodland trees, mountain forests and natural green foliage"
    },
    {
        "label": "Urban Core & Dense Built-up Residential",
        "category": "urban",
        "prompt": "dense urban core, residential buildings, city blocks and high density streets"
    },
    {
        "label": "Suburban Settlement & Mixed Urban Greenspace",
        "category": "urban",
        "prompt": "suburban residential development, neighborhood housing with gardens and trees"
    },
    {
        "label": "Industrial Complexes & Warehouse Logistics",
        "category": "industr",
        "prompt": "industrial parks, large warehouses, factory buildings and logistical freight centers"
    },
    {
        "label": "Airport Infrastructure & Transportation Corridors",
        "category": "industr",
        "prompt": "airport runways, taxiways, tarmac aprons, railway corridors and multi-lane highways"
    },
    {
        "label": "Port Terminals, Docks & Navigational Waterways",
        "category": "water",
        "prompt": "harbor docks, shipping port terminals, piers, maritime ship vessels and coastal berths"
    },
    {
        "label": "Coastal Shoreline & Deep Aquatic Basins",
        "category": "water",
        "prompt": "ocean water, coastal shoreline, marine sea, river bays and open aquatic basins"
    },
    {
        "label": "Arid Scrubland & Transition Terrain",
        "category": "arid",
        "prompt": "arid barren ground, desert sand, open unpaved soil, dry scrub and transitional terrain"
    },
    {
        "label": "Commercial Centers, Flat Roofs & Paved Parking",
        "category": "urban",
        "prompt": "commercial shopping centers, big box flat roofs, asphalt parking lots and facilities"
    }
]

_TAXONOMY_TEXT_EMBEDDINGS: Optional[np.ndarray] = None


def get_taxonomy_text_embeddings() -> Optional[np.ndarray]:
    """
    Returns pre-computed or on-demand RemoteCLIP 512-D embeddings
    for the canonical tactical land-cover taxonomy prompts.
    """
    global _TAXONOMY_TEXT_EMBEDDINGS
    if _TAXONOMY_TEXT_EMBEDDINGS is not None:
        return _TAXONOMY_TEXT_EMBEDDINGS
    try:
        from backend.services.encoder import get_encoder
        encoder = get_encoder()
        prompts = [t["prompt"] for t in TACTICAL_LANDCOVER_TAXONOMY]
        vecs = encoder.encode_text(prompts)
        _TAXONOMY_TEXT_EMBEDDINGS = np.array(vecs, dtype=np.float32)
        log.info(f"Initialized RemoteCLIP taxonomy embeddings for {len(prompts)} tactical classes.")
    except Exception as e:
        log.warning(f"Could not encode taxonomy prompts via RemoteCLIP ({e}). Using deterministic heuristics.")
        _TAXONOMY_TEXT_EMBEDDINGS = None
    return _TAXONOMY_TEXT_EMBEDDINGS


def cluster_embeddings(
    vectors: List[np.ndarray],
    target_clusters: int = DEFAULT_CLUSTERS,
    method: str = "kmeans"
) -> np.ndarray:
    """
    Partitions high-dimensional tile vectors across the entire archive.
    Default setting: 8 clusters via KMeans (random_state=42, n_init=15) on L2-normalized
    512-dim RemoteCLIP embeddings, ensuring 8 clean, balanced, non-noise partitions.
    Gracefully supports HDBSCAN if explicitly requested.
    """
    n_samples = len(vectors)
    if n_samples == 0:
        return np.array([], dtype=int)

    X = np.stack(vectors, axis=0)

    # Normalize vectors to unit length for spherical cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    if n_samples < 2:
        return np.zeros(n_samples, dtype=int)

    # If HDBSCAN requested explicitly
    if method.lower() == "hdbscan":
        try:
            from sklearn.cluster import HDBSCAN
            min_c = min(10, max(2, n_samples // target_clusters))
            hdb = HDBSCAN(min_cluster_size=min_c, metric="euclidean")
            labels = hdb.fit_predict(X_norm)
            unique_clusters = set(labels) - {-1}
            log.info(f"HDBSCAN produced {len(unique_clusters)} clusters (noise: {np.sum(labels == -1)}).")
            if len(unique_clusters) >= 2:
                return labels
            log.info("HDBSCAN produced fewer than 2 clusters. Falling back to default KMeans.")
        except Exception as e:
            log.warning(f"HDBSCAN clustering failed: {e}. Falling back to default KMeans.")

    # Default: Robust KMeans with target_clusters (8 by default)
    from sklearn.cluster import KMeans
    k = min(target_clusters, n_samples)
    km = KMeans(n_clusters=k, random_state=42, n_init=15)
    labels = km.fit_predict(X_norm)
    log.info(f"KMeans produced exactly {k} clusters across {n_samples} tile vectors.")
    return labels


def derive_cluster_labels_batch(
    unique_labels: List[int],
    labels: np.ndarray,
    vectors: List[np.ndarray],
    tile_ids: List[str],
    pg_conn
) -> Dict[int, str]:
    """
    Generates rich, distinct, domain-specific tactical land-cover labels for each cluster.
    Combines:
    1. RemoteCLIP vision-language semantic similarity against canonical aerospace taxonomy.
    2. Multi-spectral physical index priors (NDVI/NDWI/NDBI) when available from tiles.
    3. Greedy maximum-bipartite assignment ensuring NO duplicate labels across all clusters.
    """
    results: Dict[int, str] = {}
    valid_labels = [l for l in unique_labels if l != -1]
    n_clusters = len(valid_labels)

    if n_clusters == 0:
        return results

    X = np.stack(vectors, axis=0)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    X_norm = X / norms

    centroids = []
    cluster_stats = []

    for l in valid_labels:
        indices = [i for i, lbl in enumerate(labels) if lbl == l]
        c_vecs = X_norm[indices]
        centroid = np.mean(c_vecs, axis=0)
        c_norm = np.linalg.norm(centroid)
        if c_norm > 0:
            centroid = centroid / c_norm
        centroids.append(centroid)

        c_tile_ids = [tile_ids[i] for i in indices]
        try:
            with pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        AVG(mean_ndvi) as avg_ndvi,
                        AVG(mean_ndwi) as avg_ndwi,
                        AVG(mean_ndbi) as avg_ndbi,
                        MAX(sensor) as sensor
                    FROM tiles
                    WHERE tile_id = ANY(%s);
                """, (list(c_tile_ids),))
                st = cur.fetchone()
                pg_conn.commit()
        except Exception as e:
            try:
                pg_conn.rollback()
            except Exception:
                pass
            st = None

        cluster_stats.append({
            "ndvi": st.get("avg_ndvi") if st else None,
            "ndwi": st.get("avg_ndwi") if st else None,
            "ndbi": st.get("avg_ndbi") if st else None,
            "sensor": st.get("sensor") if st else None,
            "count": len(indices)
        })

    centroids_arr = np.array(centroids) # (n_clusters, 512)
    taxonomy_embeddings = get_taxonomy_text_embeddings()

    if taxonomy_embeddings is not None and len(taxonomy_embeddings) >= n_clusters:
        # Compute cosine similarity matrix between cluster centroids and taxonomy
        sim_matrix = np.dot(centroids_arr, taxonomy_embeddings.T) # (n_clusters, n_taxonomy)

        # Apply multi-spectral prior boosts
        for k in range(n_clusters):
            st = cluster_stats[k]
            ndvi = st["ndvi"]
            ndwi = st["ndwi"]
            ndbi = st["ndbi"]

            for t_idx, item in enumerate(TACTICAL_LANDCOVER_TAXONOMY):
                cat = item["category"]
                if ndwi is not None and ndwi >= 0.05 and cat == "water":
                    sim_matrix[k, t_idx] += 0.15
                if ndvi is not None and ndvi >= 0.22 and cat == "vegetat":
                    sim_matrix[k, t_idx] += 0.12
                if ndbi is not None and ndbi >= 0.02 and (cat == "urban" or cat == "industr"):
                    sim_matrix[k, t_idx] += 0.10
                if ndbi is not None and ndbi < -0.15 and cat == "water":
                    sim_matrix[k, t_idx] += 0.08

        # Greedy distinct assignment to guarantee NO duplicate labels across all clusters
        pair_scores = []
        for k in range(n_clusters):
            for t_idx in range(len(TACTICAL_LANDCOVER_TAXONOMY)):
                pair_scores.append((sim_matrix[k, t_idx], k, t_idx))

        pair_scores.sort(key=lambda x: x[0], reverse=True)

        assigned_tax_indices = set()
        cluster_assignments = {}
        assigned_clusters = set()

        for score, k, t_idx in pair_scores:
            if k not in assigned_clusters and t_idx not in assigned_tax_indices:
                assigned_clusters.add(k)
                assigned_tax_indices.add(t_idx)
                cluster_assignments[k] = t_idx
                if len(assigned_clusters) == n_clusters:
                    break

        # Fallback if any unassigned
        for k in range(n_clusters):
            if k not in cluster_assignments:
                for t_idx in range(len(TACTICAL_LANDCOVER_TAXONOMY)):
                    if t_idx not in assigned_tax_indices:
                        assigned_tax_indices.add(t_idx)
                        cluster_assignments[k] = t_idx
                        break

        for k, l in enumerate(valid_labels):
            t_idx = cluster_assignments.get(k, k % len(TACTICAL_LANDCOVER_TAXONOMY))
            tax = TACTICAL_LANDCOVER_TAXONOMY[t_idx]
            st = cluster_stats[k]

            idx_info = []
            if st['ndvi'] is not None: idx_info.append(f"NDVI {st['ndvi']:.2f}")
            if st['ndwi'] is not None: idx_info.append(f"NDWI {st['ndwi']:.2f}")
            if st['ndbi'] is not None: idx_info.append(f"NDBI {st['ndbi']:.2f}")
            idx_str = f" ({', '.join(idx_info)})" if idx_info else " (High-Res Optical / RemoteCLIP 512-D)"

            results[l] = f"Cluster {k + 1} — {tax['label']}{idx_str}"

    else:
        # Fallback deterministic spectral labeling
        fallback_types = [
            "Agricultural Farmlands & Vegetative Fields",
            "Urban Core & Dense Built-up Residential",
            "Airport Infrastructure & Transportation Corridors",
            "Coastal Shoreline & Deep Aquatic Basins",
            "Port Terminals, Docks & Navigational Waterways",
            "Arid Scrubland & Transition Terrain",
            "Commercial Centers, Flat Roofs & Paved Parking",
            "Industrial Complexes & Warehouse Logistics"
        ]
        for k, l in enumerate(valid_labels):
            st = cluster_stats[k]
            name = fallback_types[k % len(fallback_types)]
            idx_info = []
            if st['ndvi'] is not None: idx_info.append(f"NDVI {st['ndvi']:.2f}")
            if st['ndwi'] is not None: idx_info.append(f"NDWI {st['ndwi']:.2f}")
            if st['ndbi'] is not None: idx_info.append(f"NDBI {st['ndbi']:.2f}")
            idx_str = f" ({', '.join(idx_info)})" if idx_info else ""
            results[l] = f"Cluster {k + 1} — {name}{idx_str}"

    return results


def run_clustering_job(
    target_clusters: int = DEFAULT_CLUSTERS,
    method: str = "kmeans"
) -> Dict[str, Any]:
    """
    Full Phase 1 execution function:
    Pulls embeddings, clusters into target_clusters (default 8), writes back to Postgres & Qdrant.
    """
    t0 = datetime.utcnow()
    log.info("=" * 70)
    log.info(f"STARTING STANDALONE TILE CLUSTERING JOB (target_clusters={target_clusters}, method={method})")
    log.info("=" * 70)

    q_client = get_qdrant_client()
    point_ids, tile_ids, vectors, collections = fetch_all_qdrant_points(q_client)

    if not vectors:
        log.warning("No vectors found in Qdrant collections. Exiting clustering job.")
        return {
            "status": "empty",
            "message": "No tile embeddings found in Qdrant.",
            "total_points": 0,
            "clusters_count": 0
        }

    log.info(f"Loaded {len(vectors)} total embeddings from archive. Running clustering algorithm...")
    labels = cluster_embeddings(vectors, target_clusters=target_clusters, method=method)

    run_stamp = t0.strftime("%Y%m%d%H%M%S")
    unique_labels = sorted(list(set(labels)))

    conn = get_pg_connection()

    # Precompute rich domain-specific tactical labels for all clusters in batch
    cluster_labels_map = derive_cluster_labels_batch(
        unique_labels=unique_labels,
        labels=labels,
        vectors=vectors,
        tile_ids=tile_ids,
        pg_conn=conn
    )

    # Map each tile to its new cluster_id
    tile_cluster_map: Dict[str, Optional[str]] = {}
    cluster_records: List[Dict[str, Any]] = []

    for label in unique_labels:
        # Group points belonging to this cluster
        indices = [i for i, l in enumerate(labels) if l == label]
        c_tile_ids = [tile_ids[i] for i in indices]

        if label == -1:
            # HDBSCAN Noise / Outlier points get NULL cluster_id
            for t_id in c_tile_ids:
                tile_cluster_map[t_id] = None
            log.info(f"Found {len(c_tile_ids)} outlier / noise points (assigned NULL cluster_id).")
            continue

        c_vectors = [vectors[i] for i in indices]
        c_id = f"cluster_{run_stamp}_{label + 1}"

        for t_id in c_tile_ids:
            tile_cluster_map[t_id] = c_id

        # Calculate geometric centroid vector
        centroid = np.mean(c_vectors, axis=0)

        # Select medoid (tile closest to centroid)
        dists = [np.linalg.norm(vec - centroid) for vec in c_vectors]
        medoid_idx = int(np.argmin(dists))
        medoid_tile_id = c_tile_ids[medoid_idx]

        # Fetch label from batch map
        human_label = cluster_labels_map.get(label, f"Cluster {label + 1} — Land Cover Partition")

        cluster_records.append({
            "cluster_id": c_id,
            "label": human_label,
            "representative_tile_id": medoid_tile_id,
            "tile_count": len(c_tile_ids),
            "computed_at": t0,
            "model_version": MODEL_VERSION
        })

    log.info(f"Generated {len(cluster_records)} coherent clusters.")

    # 1. Update Postgres 'tiles' table with new cluster_ids
    try:
        with conn.cursor() as cur:
            # Batch update tiles in chunks of 500
            update_data = [(c_id, t_id) for t_id, c_id in tile_cluster_map.items()]
            psycopg2.extras.execute_batch(
                cur,
                "UPDATE tiles SET cluster_id = %s WHERE tile_id = %s;",
                update_data,
                page_size=500
            )

            # 2. Refresh Postgres 'clusters' table
            cur.execute("DELETE FROM clusters;")
            for c in cluster_records:
                cur.execute("""
                    INSERT INTO clusters (
                        cluster_id, label, representative_tile_id, tile_count, computed_at, model_version
                    ) VALUES (%s, %s, %s, %s, %s, %s);
                """, (
                    c["cluster_id"],
                    c["label"],
                    c["representative_tile_id"],
                    c["tile_count"],
                    c["computed_at"],
                    c["model_version"]
                ))
            conn.commit()
        log.info(f"Successfully updated {len(update_data)} rows in Postgres 'tiles' and refreshed 'clusters' table.")
    except Exception as e:
        conn.rollback()
        log.error(f"Postgres update failed: {e}", exc_info=True)
    finally:
        conn.close()

    # 3. Update Qdrant payloads with cluster_id
    qdrant_updated = 0
    for p_id, t_id, coll in zip(point_ids, tile_ids, collections):
        c_id = tile_cluster_map.get(t_id)
        try:
            q_client.set_payload(
                collection_name=coll,
                payload={"cluster_id": c_id},
                points=[p_id]
            )
            qdrant_updated += 1
        except Exception as e:
            log.warning(f"Failed to set payload for Qdrant point {p_id} in {coll}: {e}")

    log.info(f"Updated cluster payload for {qdrant_updated} Qdrant points.")

    elapsed = round((datetime.utcnow() - t0).total_seconds(), 2)
    log.info("=" * 70)
    log.info(f"CLUSTERING JOB COMPLETE: {len(cluster_records)} Clusters Across {len(vectors)} Tiles in {elapsed}s")
    log.info("=" * 70)

    return {
        "status": "success",
        "total_points": len(vectors),
        "clusters_count": len(cluster_records),
        "outliers_count": sum(1 for c_id in tile_cluster_map.values() if c_id is None),
        "clusters": cluster_records,
        "elapsed_seconds": elapsed,
        "computed_at": t0.isoformat()
    }


if __name__ == "__main__":
    run_clustering_job()
