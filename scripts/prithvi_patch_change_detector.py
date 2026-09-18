#!/usr/bin/env python3
"""
scripts/prithvi_patch_change_detector.py
========================================
Patch-wise Multi-Spectral Semantic Change Detection using Prithvi-EO-2.0-300M.

Workflow:
1. Load Before and After multi-band GeoTIFFs (.tif) and QA masks.
2. Divide multi-spectral imagery into an NxN spatial grid (e.g. 4x4 or 8x8).
3. Perform per-patch Quality Check (cloud/shadow bad fraction from mask).
4. Extract 1024-dim Prithvi-EO-2.0-300M embeddings for all valid multi-band patches.
5. Compute cosine distance (1 - cosine_similarity), robust null statistics (Median & MAD), and z-scores.
6. Extract and rank Change Candidate patches (anomalies with z >= z_threshold).
7. Generate side-by-side 4-panel visual analysis composite (OpenCV/PIL) and JSON report.
"""

import os
import sys
import json
import re
import argparse
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from PIL import Image
import cv2
import scipy.ndimage

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import rasterio
except ImportError:
    rasterio = None

from backend.services.prithvi_encoder import get_prithvi_encoder
from backend.services.semantic_change_classifier import SemanticChangeClassifier


def _to_rgb_preview(multiband: np.ndarray, band_descriptions: Optional[List[str]] = None) -> Image.Image:
    """
    Extracts an RGB visualization (H, W, 3) uint8 from a multi-band raster array (C, H, W).
    Uses percentile stretching for crisp visual rendering.
    """
    c, h, w = multiband.shape

    # Determine Red, Green, Blue channel indices
    r_idx, g_idx, b_idx = 0, 1, 2
    if band_descriptions:
        desc_lower = [str(d).lower() for d in band_descriptions]
        for idx, name in enumerate(desc_lower):
            if "red" in name or name == "b04" or name == "b4":
                r_idx = idx
            elif "green" in name or name == "b03" or name == "b3":
                g_idx = idx
            elif "blue" in name or name == "b02" or name == "b2":
                b_idx = idx
    elif c >= 4:
        # Standard Sentinel-2 L2A order in pipeline: [B02/Blue, B03/Green, B04/Red, B08/NIR]
        # Or [B04/Red, B03/Green, B02/Blue]
        b_idx, g_idx, r_idx = 0, 1, 2

    # Clamp indices
    r_idx = min(r_idx, c - 1)
    g_idx = min(g_idx, c - 1)
    b_idx = min(b_idx, c - 1)

    r = multiband[r_idx].astype(np.float32)
    g = multiband[g_idx].astype(np.float32)
    b = multiband[b_idx].astype(np.float32)

    def stretch_band(band: np.ndarray) -> np.ndarray:
        valid = band[~np.isnan(band) & ~np.isinf(band) & (band > 0)]
        if len(valid) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        p2, p98 = np.percentile(valid, (2, 98))
        if p98 > p2:
            stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0) * 255.0
        else:
            stretched = np.clip(band * 255.0, 0, 255)
        return stretched.astype(np.uint8)

    rgb = np.stack([stretch_band(r), stretch_band(g), stretch_band(b)], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def _resolve_mask_path(tif_path: str, explicit_mask: Optional[str]) -> Optional[str]:
    """Helper to auto-discover mask.tif if not explicitly passed."""
    if explicit_mask and os.path.exists(explicit_mask):
        return explicit_mask
    
    parent_dir = os.path.dirname(tif_path)
    # Check mask.tif in same folder (common in change_staging)
    cand1 = os.path.join(parent_dir, "mask.tif")
    if os.path.exists(cand1):
        return cand1
    
    # Check <tile_id>_mask.tif in same folder (common in data/tiles)
    base_name = os.path.splitext(os.path.basename(tif_path))[0]
    cand2 = os.path.join(parent_dir, f"{base_name}_mask.tif")
    if os.path.exists(cand2):
        return cand2
        
    return None


def load_multiband_and_mask(
    tif_path: str,
    mask_path: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image, Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads multi-band GeoTIFF, generates RGB preview, and loads mask if present.
    Returns: (multiband_array, pil_rgb_preview, mask_array, metadata)
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {tif_path}")

    if rasterio is None:
        raise ImportError("Rasterio is required to read GeoTIFFs. Ensure rasterio is installed.")

    with rasterio.open(tif_path) as ds:
        multiband = ds.read().astype(np.float32)
        descriptions = list(ds.descriptions) if ds.descriptions else None
        meta = {
            "crs": str(ds.crs),
            "transform": [float(x) for x in ds.transform],
            "bounds": [float(x) for x in ds.bounds],
            "count": ds.count,
            "height": ds.height,
            "width": ds.width,
            "descriptions": descriptions
        }

    c, h, w = multiband.shape
    rgb_preview = _to_rgb_preview(multiband, descriptions)

    resolved_mask = _resolve_mask_path(tif_path, mask_path)
    mask = None
    if resolved_mask and os.path.exists(resolved_mask):
        with rasterio.open(resolved_mask) as ds:
            mask = ds.read(1)
        if mask.shape != (h, w):
            mask_img = Image.fromarray(mask).resize((w, h), Image.NEAREST)
            mask = np.array(mask_img)

    return multiband, rgb_preview, mask, meta


def analyze_prithvi_patches(
    before_tif: str,
    after_tif: str,
    before_mask_path: Optional[str] = None,
    after_mask_path: Optional[str] = None,
    grid_size: int = 8,
    quality_thresh: float = 0.20,
    z_threshold: float = 2.0,
    min_dist: float = 0.01,
    min_cluster_size: int = 2,
    output_dir: str = "data/runs/prithvi_patch_change_demo",
    batch_size: int = 16
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"🛰️  PRITHVI-EO-2.0-300M MULTI-BAND CHANGE DETECTOR")
    print(f"=======================================================")
    print(f"Before GeoTIFF: {before_tif}")
    print(f"After GeoTIFF:  {after_tif}")
    print(f"Grid Layout:    {grid_size}x{grid_size} ({grid_size*grid_size} total patches)")
    print(f"Quality Filter: Max Bad Pixel Fraction <= {quality_thresh*100:.0f}%")
    print(f"Z-Threshold:    z >= {z_threshold} AND distance >= {min_dist}")
    print(f"Spatial Filter: Connected Components (Min Cluster Size >= {min_cluster_size})")
    print(f"Noise Null:     1-Sided Left-MAD (Unpolluted Noise Dispersion)")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Multi-Band Data & Mask
    mb_before, rgb_before, mask_before, meta_b = load_multiband_and_mask(before_tif, before_mask_path)
    mb_after, rgb_after, mask_after, meta_a = load_multiband_and_mask(after_tif, after_mask_path)

    W, H = rgb_before.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    print(f"Raster Dimensions: {W}x{H} | Patch Dimensions: {patch_w}x{patch_h} px | Bands: {mb_before.shape[0]}")
    print(f"Mask T1 Available: {mask_before is not None} | Mask T2 Available: {mask_after is not None}")

    # 2. Patch Extraction & Quality Audit
    patches_meta = []
    crops_before = []
    crops_after = []
    valid_indices = []

    print("\n[Stage 1] Dividing multi-spectral raster into patches and executing Quality Check...")
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            x0 = c * patch_w
            y0 = r * patch_h
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            crop_b = mb_before[:, y0:y1, x0:x1]
            crop_a = mb_after[:, y0:y1, x0:x1]

            # Quality Check from bad masks
            bad_frac_b = 0.0
            bad_frac_a = 0.0
            if mask_before is not None:
                patch_m_b = mask_before[y0:y1, x0:x1]
                bad_frac_b = float((patch_m_b > 0).mean())
            if mask_after is not None:
                patch_m_a = mask_after[y0:y1, x0:x1]
                bad_frac_a = float((patch_m_a > 0).mean())

            # Check nodata / empty array
            is_empty_b = bool(np.mean(crop_b) < 1e-4)
            is_empty_a = bool(np.mean(crop_a) < 1e-4)

            is_valid = True
            drop_reason = None
            if bad_frac_b > quality_thresh or bad_frac_a > quality_thresh:
                is_valid = False
                drop_reason = f"Cloud/Bad pixel (T1: {bad_frac_b:.1%}, T2: {bad_frac_a:.1%})"
            elif is_empty_b or is_empty_a:
                is_valid = False
                drop_reason = "NoData / Black boundary"

            patch_info = {
                "patch_id": idx,
                "row": r,
                "col": c,
                "bbox": [x0, y0, x1, y1],
                "is_valid": is_valid,
                "drop_reason": drop_reason,
                "bad_frac_t1": round(bad_frac_b, 4),
                "bad_frac_t2": round(bad_frac_a, 4),
                "distance": 0.0,
                "z_score": 0.0,
                "cluster_id": None,
                "cluster_size": 0,
                "is_candidate": False,
                "is_isolated_noise": False
            }
            patches_meta.append(patch_info)

            if is_valid:
                valid_indices.append(idx)
                crops_before.append(crop_b)
                crops_after.append(crop_a)

    print(f"Total Patches: {len(patches_meta)} | Valid/Clear: {len(valid_indices)} | Dropped: {len(patches_meta) - len(valid_indices)}")

    # 3. Embedding Generation via Prithvi-EO-2.0-300M
    print("\n[Stage 2] Computing 1024-dim Prithvi-EO-2.0-300M embeddings for clear multi-spectral patches...")
    encoder = get_prithvi_encoder()

    if valid_indices:
        print(f"Encoding {len(crops_before)} Before multi-spectral patches...")
        vecs_b = encoder.encode_multiband_patches(crops_before, batch_size=batch_size)
        print(f"Encoding {len(crops_after)} After multi-spectral patches...")
        vecs_a = encoder.encode_multiband_patches(crops_after, batch_size=batch_size)

        # 4. Semantic Distance (1.0 - Cosine Similarity)
        sims = np.sum(vecs_b * vecs_a, axis=1)
        sims = np.clip(sims, -1.0, 1.0)
        distances = 1.0 - sims

        # Assign back to patch meta
        for i, val_idx in enumerate(valid_indices):
            patches_meta[val_idx]["distance"] = float(distances[i])

        # 5. Robust Statistics: Median and 1-Sided Left-MAD
        med_dist = float(np.median(distances))
        full_mad = float(np.median(np.abs(distances - med_dist)))
        
        # 1-Sided Left-MAD: calculate deviations ONLY from patches <= median
        # Real anomalies are in the right tail; Left-MAD prevents genuine change from inflating the threshold
        left_deviations = med_dist - distances[distances <= med_dist]
        if len(left_deviations) > 0:
            left_mad = float(np.median(left_deviations)) * 1.4826
        else:
            left_mad = full_mad
        left_mad = max(left_mad, 1e-4)

        print(f"\n[Stage 3] Robust Spatial Distance Distribution (Null Model):")
        print(f"  • Min Distance:    {float(np.min(distances)):.4f}")
        print(f"  • Median Distance: {med_dist:.4f}")
        print(f"  • Standard MAD:    {full_mad:.4f}")
        print(f"  • 1-Sided Left-MAD:{left_mad:.4f} (Used for z-scoring)")
        print(f"  • Max Distance:    {float(np.max(distances)):.4f}")

        # Compute z-scores for all valid patches using Left-MAD
        cand_grid = np.zeros((grid_size, grid_size), dtype=bool)
        for val_idx in valid_indices:
            d = patches_meta[val_idx]["distance"]
            z = (d - med_dist) / left_mad
            patches_meta[val_idx]["z_score"] = float(round(z, 2))
            if z >= z_threshold and d >= min_dist:
                r, c = patches_meta[val_idx]["row"], patches_meta[val_idx]["col"]
                cand_grid[r, c] = True

        # 6. 8-Neighbor Connected Components Clustering
        # Single isolated noisy patches are suppressed unless they form contiguous clusters >= min_cluster_size
        labeled_grid, num_clusters = scipy.ndimage.label(cand_grid, structure=np.ones((3, 3)))
        cluster_sizes = {}
        for lbl in range(1, num_clusters + 1):
            cluster_sizes[lbl] = int((labeled_grid == lbl).sum())

        candidates = []
        isolated_noise = []

        for val_idx in valid_indices:
            r, c = patches_meta[val_idx]["row"], patches_meta[val_idx]["col"]
            lbl = int(labeled_grid[r, c])
            c_size = cluster_sizes.get(lbl, 0)
            patches_meta[val_idx]["cluster_id"] = lbl if lbl > 0 else None
            patches_meta[val_idx]["cluster_size"] = c_size

            if lbl > 0:
                if c_size >= min_cluster_size:
                    patches_meta[val_idx]["is_candidate"] = True
                    candidates.append(patches_meta[val_idx])
                else:
                    patches_meta[val_idx]["is_candidate"] = False
                    patches_meta[val_idx]["is_isolated_noise"] = True
                    isolated_noise.append(patches_meta[val_idx])

        candidates.sort(key=lambda x: -x["distance"])
        isolated_noise.sort(key=lambda x: -x["distance"])

        print(f"\n[Stage 4] Spatial Clustering Analysis:")
        print(f"  • Total Anomaly Patches (z >= {z_threshold}): {len(candidates) + len(isolated_noise)}")
        print(f"  • Connected Components Identified: {num_clusters}")
        print(f"  • Confirmed Clustered Candidates (Size >= {min_cluster_size}): {len(candidates)}")
        print(f"  • Suppressed Isolated Noise Patches (Size < {min_cluster_size}): {len(isolated_noise)}")

        for rank, cand in enumerate(candidates[:10], 1):
            print(f"  Rank #{rank:02d} | Patch [{cand['row']}, {cand['col']}] | Cluster #{cand['cluster_id']} (size: {cand['cluster_size']}) | Dist: {cand['distance']:.4f} | z-score: {cand['z_score']:+.2f}")

        # Stage 4.5: Physical Land Cover Classification & Spectral Stability Rejection (NDVI, NDWI, NDBI)
        desc = meta_b.get("descriptions") or meta_a.get("descriptions")
        confirmed_candidates, dropped_stable = SemanticChangeClassifier.evaluate_candidates(
            candidates=candidates,
            mb_before=mb_before,
            mb_after=mb_after,
            band_descriptions=desc,
            stability_threshold=0.002
        )
        cluster_semantics = SemanticChangeClassifier.aggregate_cluster_consensus(confirmed_candidates)
        candidates = confirmed_candidates

        print(f"\n[Stage 4.5] Physical Land Cover Classification & Cluster Consensus:")
        print(f"  • Confirmed Spectrally Active Candidates: {len(candidates)}")
        print(f"  • Invariance Filter (Dropped Stable):    {len(dropped_stable)}")
        for cid, cinfo in cluster_semantics.items():
            print(f"  • Cluster #{cid} ({cinfo['num_patches']} patches): {cinfo['majority_class_t1']} → {cinfo['majority_class_t2']} [{cinfo['transition_type']}]")
    else:
        print("Warning: No clear/valid patches found after quality filtering!")
        candidates = []
        isolated_noise = []
        dropped_stable = []
        cluster_semantics = {}
        med_dist, left_mad, full_mad = 0.0, 1.0, 1.0
        num_clusters = 0

    # 7. Generate 4-Panel Side-by-Side Visual Plot
    print("\n[Stage 5] Generating 4-panel visual analysis panel...")
    p1 = np.array(rgb_before).copy()
    p2 = np.array(rgb_after).copy()
    p4 = np.array(rgb_after).copy()

    # Draw grid lines on p1, p2, p4
    for p in [p1, p2, p4]:
        for i in range(1, grid_size):
            cv2.line(p, (i * patch_w, 0), (i * patch_w, H), (56, 189, 248), 1, cv2.LINE_AA)
            cv2.line(p, (0, i * patch_h), (W, i * patch_h), (56, 189, 248), 1, cv2.LINE_AA)

    # Panel 3: Semantic Heatmap
    min_d = float(np.min(distances)) if valid_indices else 0.0
    max_d = float(np.max(distances)) if valid_indices else 1.0
    d_range = max(max_d - min_d, 1e-4)

    dist_grid_norm = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for p in patches_meta:
        r, c = p["row"], p["col"]
        if p["is_valid"]:
            norm_val = int(255.0 * (p["distance"] - min_d) / d_range)
            dist_grid_norm[r, c] = np.clip(norm_val, 0, 255)
        else:
            dist_grid_norm[r, c] = 0

    heat_small = cv2.applyColorMap(dist_grid_norm, cv2.COLORMAP_MAGMA)
    heat_small = cv2.cvtColor(heat_small, cv2.COLOR_BGR2RGB)
    p3 = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # Darken dropped patches in heatmap
    for p in patches_meta:
        if not p["is_valid"]:
            x0, y0, x1, y1 = p["bbox"]
            p3[y0:y1, x0:x1] = (50, 50, 50)
            cv2.line(p3, (x0, y0), (x1, y1), (100, 100, 100), 1)
            cv2.line(p3, (x0, y1), (x1, y0), (100, 100, 100), 1)

    for i in range(1, grid_size):
        cv2.line(p3, (i * patch_w, 0), (i * patch_w, H), (30, 41, 59), 1)
        cv2.line(p3, (0, i * patch_h), (W, i * patch_h), (30, 41, 59), 1)

    # Draw Suppressed Isolated Noise Patches with dashed/cyan outline
    for noise_patch in isolated_noise:
        x0, y0, x1, y1 = noise_patch["bbox"]
        cv2.rectangle(p4, (x0 + 2, y0 + 2), (x1 - 2, y1 - 2), (148, 163, 184), 1)
        cv2.putText(p4, "noise", (x0 + 4, y0 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (148, 163, 184), 1, cv2.LINE_AA)

    # Highlight Confirmed Connected Candidates on Panel 4
    for rank, cand in enumerate(candidates, 1):
        x0, y0, x1, y1 = cand["bbox"]
        color = (239, 68, 68) if rank <= 3 else (245, 158, 11)
        cv2.rectangle(p4, (x0, y0), (x1, y1), color, 2)
        label = f"#{rank} z={cand['z_score']:+.1f} [C{cand['cluster_id']}]"
        badge_w = 95
        cv2.rectangle(p4, (x0, y0), (x0 + badge_w, y0 + 16), color, -1)
        cv2.putText(p4, label, (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)

    # Composite canvas
    header_h = 50
    spacing = 15
    canvas_w = (W * 4) + (spacing * 5)
    canvas_h = H + header_h + 30
    canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)  # Slate dark #0f172a

    def extract_date_label(path_str: str) -> str:
        m1 = re.search(r"(\d{4}-\d{2}-\d{2})", path_str)
        if m1:
            return m1.group(1)
        m2 = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(path_str))
        if m2:
            return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"
        return os.path.basename(path_str)

    date_b = extract_date_label(before_tif)
    date_a = extract_date_label(after_tif)

    titles = [
        f"T1 Before ({date_b})",
        f"T2 After ({date_a})",
        f"Prithvi 300M Dist (Med:{med_dist:.3f}, Left-MAD:{left_mad:.3f})",
        f"Change Clusters ({len(candidates)} confirmed, {len(isolated_noise)} noise dropped)"
    ]

    for idx, (p_img, title) in enumerate(zip([p1, p2, p3, p4], titles)):
        x_offset = spacing + idx * (W + spacing)
        y_offset = header_h
        canvas[y_offset : y_offset + H, x_offset : x_offset + W] = p_img
        cv2.putText(canvas, title, (x_offset, header_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

    plot_out = os.path.join(output_dir, "patch_change_analysis.png")
    plot_out_c = os.path.join(output_dir, "c.png")
    Image.fromarray(canvas).save(plot_out)
    Image.fromarray(canvas).save(plot_out_c)
    print(f"Visual Analysis Plot saved to: {plot_out}")
    print(f"Visual Analysis Plot (c.png) saved to: {plot_out_c}")

    # 8. Save JSON Result matching downstream contract
    result_data = {
        "encoder": "Prithvi-EO-2.0-300M",
        "embedding_dim": 1024,
        "before_image": before_tif,
        "after_image": after_tif,
        "grid_size": grid_size,
        "patch_pixels": [patch_w, patch_h],
        "total_patches": len(patches_meta),
        "valid_patches": len(valid_indices),
        "dropped_patches": len(patches_meta) - len(valid_indices),
        "robust_stats": {
            "median_distance": round(med_dist, 4),
            "left_mad": round(left_mad, 4),
            "full_mad": round(full_mad, 4),
            "z_threshold": z_threshold,
            "min_cluster_size": min_cluster_size,
            "total_clusters_found": num_clusters
        },
        "num_candidates": len(candidates),
        "num_isolated_noise": len(isolated_noise),
        "num_spectral_stable_dropped": len(dropped_stable),
        "candidates": candidates,
        "isolated_noise": isolated_noise,
        "spectral_stable_dropped": dropped_stable,
        "cluster_semantics": cluster_semantics,
        "all_patches": patches_meta
    }

    json_out = os.path.join(output_dir, "patch_change_results.json")
    with open(json_out, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Detailed JSON results saved to: {json_out}")
    print(f"=======================================================\n")
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-spectral patch change detection using Prithvi-EO-2.0-300M")
    parser.add_argument(
        "--before",
        required=True,
        help="Path to Before GeoTIFF (.tif)"
    )
    parser.add_argument(
        "--after",
        required=True,
        help="Path to After GeoTIFF (.tif)"
    )
    parser.add_argument(
        "--before-mask",
        default=None,
        help="Path to Before QA mask GeoTIFF (optional: auto-discovered if sibling mask.tif exists)"
    )
    parser.add_argument(
        "--after-mask",
        default=None,
        help="Path to After QA mask GeoTIFF (optional: auto-discovered if sibling mask.tif exists)"
    )
    parser.add_argument("--grid", type=int, default=8, help="Grid size (default 8 for 8x8 = 64 grids, 64x64 px each)")
    parser.add_argument("--quality-thresh", type=float, default=0.20, help="Max bad pixel fraction per patch")
    parser.add_argument("--z-thresh", type=float, default=2.0, help="Z-score threshold for change candidates")
    parser.add_argument("--min-dist", type=float, default=0.01, help="Minimum cosine distance required for change candidate (default 0.01)")
    parser.add_argument("--min-cluster", type=int, default=2, help="Minimum connected patches required to form a confirmed change cluster (default 2)")
    parser.add_argument("--out-dir", default="data/runs/prithvi_patch_change_demo", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")

    args = parser.parse_args()

    analyze_prithvi_patches(
        before_tif=args.before,
        after_tif=args.after,
        before_mask_path=args.before_mask,
        after_mask_path=args.after_mask,
        grid_size=args.grid,
        quality_thresh=args.quality_thresh,
        z_threshold=args.z_thresh,
        min_dist=args.min_dist,
        min_cluster_size=args.min_cluster,
        output_dir=args.out_dir,
        batch_size=args.batch_size
    )
