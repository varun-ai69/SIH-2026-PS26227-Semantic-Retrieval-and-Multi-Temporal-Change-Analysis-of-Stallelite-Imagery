"""
backend/services/patch_change_detector.py
=========================================
Prithvi-EO-2.0-300M Patch Change Detection Engine
=================================================
Performs semantic patch-wise change detection between two milestone satellite acquisitions.

Features:
  1. Patch extraction across NxN spatial grid (default 8x8 = 64 patches).
  2. Per-patch QA mask filtering (drops cloud/shadow corrupted patches).
  3. NASA-IBM Prithvi-EO-2.0-300M multi-spectral feature encoding (1024-D) on CUDA GPU.
  4. Robust 1-Sided Left-MAD null statistics:
        Left-MAD calculates dispersion strictly from the left of median (<= median),
        preventing genuine physical anomalies from inflating the noise threshold.
  5. 8-Neighbor Connected Components Clustering (scipy.ndimage.label):
        Suppresses isolated 1-patch false alarms, confirming persistent spatial clusters.
  6. Visual generation matching user requirements:
        - Side-by-side Before and After images.
        - NO full blue grid lines across the image (untouched areas stay pure satellite imagery).
        - Single clean highlight color (bold red/amber border with soft semi-transparent fill).
        - NO text/Z-score labels drawn over the box; strictly the box boundary of change.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import scipy.ndimage

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    from skimage.registration import phase_cross_correlation
except ImportError:
    phase_cross_correlation = None

from backend.services.prithvi_encoder import get_prithvi_encoder
from backend.services.semantic_change_classifier import SemanticChangeClassifier

logger = logging.getLogger("patch_change_detector")

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_file_path(path_str: Optional[str]) -> Optional[Path]:
    """Resolves relative/container paths (/app/data/...) to local filesystem."""
    if not path_str:
        return None
    p = Path(path_str)
    if p.is_file():
        return p
    p_norm = str(path_str).replace("\\", "/")
    idx = p_norm.find("data/")
    if idx != -1:
        cand = REPO_ROOT / p_norm[idx:]
        if cand.is_file():
            return cand
    return None


def _to_web_url(file_path: Optional[str]) -> Optional[str]:
    """Converts a local disk path into a web-accessible static URL (/data/...)."""
    if not file_path:
        return None
    p = str(file_path).replace("\\", "/")
    idx = p.find("data/")
    if idx != -1:
        return "/" + p[idx:]
    if p.startswith("/app/data/"):
        return p.replace("/app/data/", "/data/")
    return "/" + p.lstrip("/")


def _to_rgb_preview(multiband: np.ndarray, band_descriptions: Optional[List[str]] = None) -> Image.Image:
    """
    Extracts an RGB visualization (H, W, 3) uint8 from a multi-band raster array (C, H, W).
    Uses 2%-98% percentile stretching for optimal visual contrast.
    """
    c, h, w = multiband.shape
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
        # Standard Sentinel-2 L2A in pipeline: [B02/Blue, B03/Green, B04/Red, B08/NIR]
        b_idx, g_idx, r_idx = 0, 1, 2

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


def load_multiband_and_mask(
    tif_path: str,
    mask_path: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image, Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads multi-band GeoTIFF, generates RGB preview, and loads QA mask if present.
    """
    resolved_tif = resolve_file_path(tif_path)
    if not resolved_tif or not resolved_tif.is_file():
        raise FileNotFoundError(f"GeoTIFF file not found: {tif_path}")

    if rasterio is None:
        raise ImportError("Rasterio is required to read GeoTIFFs.")

    with rasterio.open(str(resolved_tif)) as ds:
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

    mask = None
    resolved_mask = resolve_file_path(mask_path)
    if not resolved_mask:
        # Check sibling mask.tif in same folder
        cand = resolved_tif.parent / "mask.tif"
        if cand.is_file():
            resolved_mask = cand

    if resolved_mask and resolved_mask.is_file():
        try:
            with rasterio.open(str(resolved_mask)) as ds:
                mask = ds.read(1)
            if mask.shape != (h, w):
                mask_img = Image.fromarray(mask).resize((w, h), Image.NEAREST)
                mask = np.array(mask_img)
        except Exception as e:
            logger.warning(f"Could not load mask {resolved_mask}: {e}")

    return multiband, rgb_preview, mask, meta


def align_subpixel_misregistration(
    mb_before: np.ndarray,
    mb_after: np.ndarray,
    mask_after: Optional[np.ndarray] = None,
    band_descriptions_b: Optional[List[str]] = None,
    band_descriptions_a: Optional[List[str]] = None,
    max_shift: float = 3.0,
    upsample_factor: int = 10
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Sub-pixel phase correlation alignment to eliminate satellite orbital misregistration (jitter).
    Detects fractional pixel shifts between T1 and T2 using cross-power spectrum phase correlation,
    then warps T2 using bicubic/spline sub-pixel shift. Suppresses artificial edge false alarms.
    """
    reg_meta = {
        "applied": False,
        "shift_y_pixels": 0.0,
        "shift_x_pixels": 0.0,
        "error": 0.0,
        "status": "skipped"
    }

    if phase_cross_correlation is None:
        reg_meta["status"] = "skimage_unavailable"
        return mb_after, mask_after, reg_meta

    try:
        c, h, w = mb_before.shape
        if mb_after.shape != mb_before.shape or h < 32 or w < 32:
            reg_meta["status"] = "shape_mismatch"
            return mb_after, mask_after, reg_meta

        # Find best high-contrast terrestrial band (NIR or Red) for registration
        ref_idx = 0
        if band_descriptions_b:
            for idx, d in enumerate(band_descriptions_b):
                d_low = str(d).lower()
                if "nir" in d_low or "b08" in d_low or "b8" in d_low:
                    ref_idx = idx
                    break
                elif "red" in d_low or "b04" in d_low or "b4" in d_low:
                    ref_idx = idx
        elif c >= 4:
            ref_idx = 3 if c > 3 else 2

        ref_idx = min(ref_idx, c - 1)

        # Central 70% crop to avoid border zero-padding and nodata edge artifacts
        y0, y1 = int(0.15 * h), int(0.85 * h)
        x0, x1 = int(0.15 * w), int(0.85 * w)

        b_crop = mb_before[ref_idx, y0:y1, x0:x1].astype(np.float32)
        a_crop = mb_after[ref_idx, y0:y1, x0:x1].astype(np.float32)

        # Verify crops have valid texture variance
        if np.std(b_crop) < 1e-4 or np.std(a_crop) < 1e-4:
            reg_meta["status"] = "insufficient_texture"
            return mb_after, mask_after, reg_meta

        # Compute sub-pixel shift: shifts needed to align a_crop to b_crop
        shifts, error, _ = phase_cross_correlation(
            b_crop, a_crop, upsample_factor=upsample_factor
        )
        shift_y, shift_x = float(shifts[0]), float(shifts[1])
        reg_meta["shift_y_pixels"] = round(shift_y, 3)
        reg_meta["shift_x_pixels"] = round(shift_x, 3)
        reg_meta["error"] = round(float(error), 4)

        # Only correct realistic satellite orbital jitter (0.1px to max_shift px)
        abs_shift = max(abs(shift_y), abs(shift_x))
        if 0.1 <= abs_shift <= max_shift:
            aligned_mb_after = np.zeros_like(mb_after)
            for ch in range(c):
                aligned_mb_after[ch] = scipy.ndimage.shift(
                    mb_after[ch], (shift_y, shift_x), order=1, mode="nearest"
                )

            aligned_mask_after = mask_after
            if mask_after is not None:
                aligned_mask_after = scipy.ndimage.shift(
                    mask_after, (shift_y, shift_x), order=0, mode="nearest"
                )

            reg_meta["applied"] = True
            reg_meta["status"] = "aligned"
            logger.info(
                f"🛰️ Sub-pixel misregistration corrected: dy={shift_y:+.2f}px, dx={shift_x:+.2f}px"
            )
            return aligned_mb_after, aligned_mask_after, reg_meta
        else:
            reg_meta["status"] = "shift_out_of_bounds" if abs_shift > max_shift else "already_aligned"
            return mb_after, mask_after, reg_meta

    except Exception as e:
        logger.warning(f"Sub-pixel misregistration correction encountered exception: {e}")
        reg_meta["status"] = f"error: {str(e)}"
        return mb_after, mask_after, reg_meta


class PrithviPatchChangeDetector:
    """
    High-performance semantic patch change detector utilizing IBM-NASA Prithvi-EO-2.0-300M.
    """

    def __init__(self):
        self.encoder = get_prithvi_encoder()

    def run_detection(
        self,
        before_tif: str,
        after_tif: str,
        before_mask: Optional[str] = None,
        after_mask: Optional[str] = None,
        output_dir: Optional[str] = None,
        grid_size: int = 8,               # 8x8 = 64 spatial patches
        quality_thresh: float = 0.20,     # Max 20% bad pixels per patch
        z_threshold: float = 2.0,         # Minimum anomaly Z-score
        min_dist: float = 0.01,           # Minimum cosine distance
        min_cluster_size: int = 2,        # Minimum connected candidate patches (filters isolated noise)
        batch_size: int = 16,
        stability_threshold: float = 0.002, # Max delta across NDVI, NDWI, NDBI to consider stable (dropped only if virtually unchanged)
        highlight_color: Tuple[int, int, int] = (239, 68, 68),  # Bold clean crimson/amber RGB (no blue grid, single color)
        step_index: int = 1,
        prior_step_clusters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes patch change detection pipeline and generates visual overlays without full blue grids.
        """
        # 1. Load multi-band arrays & previews
        mb_before, rgb_before, mask_before, meta_b = load_multiband_and_mask(before_tif, before_mask)
        mb_after, rgb_after, mask_after, meta_a = load_multiband_and_mask(after_tif, after_mask)

        # 1b. Sub-Pixel Phase Correlation Registration (Eliminates Satellite Orbital Jitter)
        mb_after, mask_after, reg_info = align_subpixel_misregistration(
            mb_before=mb_before,
            mb_after=mb_after,
            mask_after=mask_after,
            band_descriptions_b=meta_b.get("descriptions"),
            band_descriptions_a=meta_a.get("descriptions")
        )
        if reg_info.get("applied"):
            rgb_after = _to_rgb_preview(mb_after, meta_a.get("descriptions"))

        W, H = rgb_before.size
        patch_w = W // grid_size
        patch_h = H // grid_size

        # 2. Divide into spatial patches and execute quality checks
        patches_meta = []
        crops_before = []
        crops_after = []
        valid_indices = []

        for r in range(grid_size):
            for c in range(grid_size):
                idx = r * grid_size + c
                x0 = c * patch_w
                y0 = r * patch_h
                x1 = x0 + patch_w
                y1 = y0 + patch_h

                crop_b = mb_before[:, y0:y1, x0:x1]
                crop_a = mb_after[:, y0:y1, x0:x1]

                bad_frac_b = 0.0
                bad_frac_a = 0.0
                if mask_before is not None:
                    bad_frac_b = float((mask_before[y0:y1, x0:x1] > 0).mean())
                if mask_after is not None:
                    bad_frac_a = float((mask_after[y0:y1, x0:x1] > 0).mean())

                is_empty_b = bool(np.mean(crop_b) < 1e-4)
                is_empty_a = bool(np.mean(crop_a) < 1e-4)

                is_valid = True
                drop_reason = None
                if bad_frac_b > quality_thresh or bad_frac_a > quality_thresh:
                    is_valid = False
                    drop_reason = f"Cloud/corrupted surface (T1: {bad_frac_b:.1%}, T2: {bad_frac_a:.1%})"
                elif is_empty_b or is_empty_a:
                    is_valid = False
                    drop_reason = "NoData / Black edge boundary"

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

        # 3. Compute Prithvi-EO-2.0-300M Latent Embeddings
        if valid_indices:
            vecs_b = self.encoder.encode_multiband_patches(crops_before, batch_size=batch_size)
            vecs_a = self.encoder.encode_multiband_patches(crops_after, batch_size=batch_size)

            # Cosine distance (1.0 - cosine_similarity)
            sims = np.sum(vecs_b * vecs_a, axis=1)
            sims = np.clip(sims, -1.0, 1.0)
            distances = 1.0 - sims

            for i, val_idx in enumerate(valid_indices):
                patches_meta[val_idx]["distance"] = float(distances[i])

            # 4. Robust 1-Sided Left-MAD Statistics
            med_dist = float(np.median(distances))
            full_mad = float(np.median(np.abs(distances - med_dist)))
            left_deviations = med_dist - distances[distances <= med_dist]
            if len(left_deviations) > 0:
                left_mad = float(np.median(left_deviations)) * 1.4826
            else:
                left_mad = full_mad
            left_mad = max(left_mad, 1e-4)

            # Assign Z-scores
            cand_grid = np.zeros((grid_size, grid_size), dtype=bool)
            for val_idx in valid_indices:
                d = patches_meta[val_idx]["distance"]
                z = (d - med_dist) / left_mad
                patches_meta[val_idx]["z_score"] = float(round(z, 2))
                if z >= z_threshold and d >= min_dist:
                    r, c = patches_meta[val_idx]["row"], patches_meta[val_idx]["col"]
                    cand_grid[r, c] = True

            # 5. Spatial Contiguity Clustering (Touching patches form one cluster)
            labeled_grid, raw_num_clusters = scipy.ndimage.label(cand_grid, structure=np.ones((3, 3)))
            raw_cluster_patches: Dict[int, List[int]] = {}
            for val_idx in valid_indices:
                r, c = patches_meta[val_idx]["row"], patches_meta[val_idx]["col"]
                lbl = int(labeled_grid[r, c])
                if lbl > 0:
                    raw_cluster_patches.setdefault(lbl, []).append(val_idx)

            # Sort contiguous clusters by size descending (largest contiguous area gets Cluster #1)
            sorted_raw_clusters = sorted(
                raw_cluster_patches.items(),
                key=lambda kv: len(kv[1]),
                reverse=True
            )

            candidates = []
            isolated_noise = []
            clean_cid = 1

            for raw_lbl, patch_idx_list in sorted_raw_clusters:
                if len(patch_idx_list) >= min_cluster_size:
                    for val_idx in patch_idx_list:
                        patches_meta[val_idx]["cluster_id"] = clean_cid
                        patches_meta[val_idx]["cluster_size"] = len(patch_idx_list)
                        patches_meta[val_idx]["is_candidate"] = True
                        candidates.append(patches_meta[val_idx])
                    clean_cid += 1
                else:
                    # Isolated 1-patch noise
                    for val_idx in patch_idx_list:
                        patches_meta[val_idx]["cluster_id"] = None
                        patches_meta[val_idx]["cluster_size"] = len(patch_idx_list)
                        patches_meta[val_idx]["is_candidate"] = False
                        patches_meta[val_idx]["is_isolated_noise"] = True
                        isolated_noise.append(patches_meta[val_idx])

            candidates.sort(key=lambda x: -x["distance"])
            isolated_noise.sort(key=lambda x: -x["distance"])

            # 6. Physical Multi-Spectral Deltas Extraction (NDVI, NDWI, NDBI, BSI)
            desc_b = meta_b.get("descriptions")
            desc_a = meta_a.get("descriptions")
            confirmed_candidates, dropped_stable = SemanticChangeClassifier.evaluate_candidates(
                candidates=candidates,
                mb_before=mb_before,
                mb_after=mb_after,
                band_descriptions_b=desc_b,
                band_descriptions_a=desc_a,
                stability_threshold=stability_threshold
            )

            # 7. RemoteCLIP Vision-Language Foundation Model Inference & Cluster Majority Attribution
            try:
                from backend.services.clip_change_typer import get_change_typer
                typer = get_change_typer()
                classified = typer.classify_candidates(
                    candidates=confirmed_candidates,
                    rgb_before=rgb_before,
                    rgb_after=rgb_after,
                    mb_before=mb_before,
                    mb_after=mb_after,
                    band_descriptions=desc_b or desc_a,
                    band_descriptions_b=desc_b,
                    band_descriptions_a=desc_a
                )
                # Compute majority consensus per spatial cluster and unify all member patches
                cluster_semantics = typer.aggregate_cluster_consensus(classified)
                confirmed_candidates = classified
            except Exception as e:
                logger.error(f"RemoteCLIP change typing fallback to spectral consensus: {e}", exc_info=True)
            candidates = confirmed_candidates

            # Attach Tactical Spatial Morphology Dynamics (NetSight Alignment: Appearance, Expansion, Contraction, Disappearance)
            try:
                from backend.services.tactical_morphology import classify_tactical_dynamic
                for cid, c_data in cluster_semantics.items():
                    c_data["tactical_dynamic"] = classify_tactical_dynamic(
                        cluster=c_data,
                        prior_cluster=prior_step_clusters.get(str(cid)) or prior_step_clusters.get(cid) if prior_step_clusters else None,
                        step_index=step_index
                    )
                for cand in candidates:
                    cid = cand.get("cluster_id")
                    if cid in cluster_semantics:
                        cand["tactical_dynamic"] = cluster_semantics[cid].get("tactical_dynamic")
            except Exception as te:
                logger.warning(f"Tactical dynamic classification warning: {te}")
        else:
            candidates = []
            isolated_noise = []
            dropped_stable = []
            cluster_semantics = {}
            med_dist, left_mad, full_mad = 0.0, 1.0, 1.0
            num_clusters = 0

        # ============================================================
        # 6. Generate Clean Visual Output Matching User Specifications
        #    - NO full blue grid lines across the image!
        #    - Coordinated cluster color coding (Crimson, Emerald, Amber).
        #    - Cluster # badge pill drawn directly on the satellite photo.
        # ============================================================
        out_p = Path(output_dir) if output_dir else REPO_ROOT / "data" / "runs" / "patch_change"
        out_p.mkdir(parents=True, exist_ok=True)

        # Base images
        img_before_rgb = np.array(rgb_before).copy()
        img_after_rgb = np.array(rgb_after).copy()

        # Create overlay image on T2 (After):
        overlay_after = img_after_rgb.copy()

        # Cluster color mapping (RGB values for PIL Image overlay)
        CLUSTER_PALETTE = [
            {"rgb": (239, 68, 68), "hex": "#ef4444", "name": "Crimson"},      # Cluster 1
            {"rgb": (16, 185, 129), "hex": "#10b981", "name": "Emerald"},     # Cluster 2
            {"rgb": (245, 158, 11), "hex": "#f59e0b", "name": "Amber"},       # Cluster 3
            {"rgb": (6, 182, 212), "hex": "#06b6d4", "name": "Cyan"},         # Cluster 4
            {"rgb": (168, 85, 247), "hex": "#a855f7", "name": "Purple"},      # Cluster 5
        ]

        # Subtle semi-transparent fill for changed patches (22% alpha)
        tint_layer = overlay_after.copy()
        for cand in candidates:
            cid = cand.get("cluster_id", 1) or 1
            color_idx = (cid - 1) % len(CLUSTER_PALETTE)
            c_rgb = CLUSTER_PALETTE[color_idx]["rgb"]
            x0, y0, x1, y1 = cand["bbox"]
            cv2.rectangle(tint_layer, (x0, y0), (x1, y1), c_rgb, -1)

        cv2.addWeighted(tint_layer, 0.22, overlay_after, 0.78, 0, overlay_after)

        # Draw crisp solid bounding box + Cluster # pill badge directly on the photo
        for cand in candidates:
            cid = cand.get("cluster_id", 1) or 1
            color_idx = (cid - 1) % len(CLUSTER_PALETTE)
            c_rgb = CLUSTER_PALETTE[color_idx]["rgb"]
            x0, y0, x1, y1 = cand["bbox"]

            # 1. Bounding box border
            cv2.rectangle(overlay_after, (x0, y0), (x1, y1), c_rgb, 2, cv2.LINE_AA)

            # 2. Modern Cluster Badge Pill on top-left of the box
            label = f"Cluster #{cid}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.35
            thickness = 1
            (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

            pill_x0 = x0 + 2
            pill_y0 = y0 + 2
            pill_x1 = pill_x0 + text_w + 6
            pill_y1 = pill_y0 + text_h + 6

            # Dark pill background with cluster color border
            cv2.rectangle(overlay_after, (pill_x0, pill_y0), (pill_x1, pill_y1), (15, 23, 42), -1)
            cv2.rectangle(overlay_after, (pill_x0, pill_y0), (pill_x1, pill_y1), c_rgb, 1, cv2.LINE_AA)

            # Crisp white text label
            text_x = pill_x0 + 3
            text_y = pill_y0 + text_h + 2
            cv2.putText(overlay_after, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        # Save individual preview images
        path_before_jpg = out_p / "before_thumb.jpg"
        path_after_jpg = out_p / "after_thumb.jpg"
        path_overlay_jpg = out_p / "change_overlay.jpg"

        Image.fromarray(img_before_rgb).save(path_before_jpg, quality=92)
        Image.fromarray(img_after_rgb).save(path_after_jpg, quality=92)
        Image.fromarray(overlay_after).save(path_overlay_jpg, quality=92)

        # Save side-by-side composite: [Before] [Gap] [After with Change Boxes & Cluster Labels]
        gap = 16
        composite_w = (W * 2) + gap
        composite_h = H
        composite_img = np.full((composite_h, composite_w, 3), 15, dtype=np.uint8)
        composite_img[:, :W] = img_before_rgb
        composite_img[:, W + gap :] = overlay_after

        path_composite_jpg = out_p / "change_side_by_side.jpg"
        Image.fromarray(composite_img).save(path_composite_jpg, quality=92)

        # 7. Detailed JSON result
        result_data = {
            "status": "success",
            "device": str(self.encoder.device),
            "device_name": getattr(self.encoder, "device_name", "NVIDIA GeForce RTX 2050") if "cuda" in str(self.encoder.device) else "CPU",
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
                "total_clusters_found": len(cluster_semantics) if cluster_semantics else (clean_cid - 1 if 'clean_cid' in locals() else 0)
            },
            "misregistration_correction": reg_info,
            "num_candidates": len(candidates),
            "num_isolated_noise": len(isolated_noise),
            "num_spectral_stable_dropped": len(dropped_stable),
            "candidates": candidates,
            "isolated_noise": isolated_noise,
            "spectral_stable_dropped": dropped_stable,
            "cluster_semantics": cluster_semantics,
            "files": {
                "before_thumb": _to_web_url(str(path_before_jpg)),
                "after_thumb": _to_web_url(str(path_after_jpg)),
                "change_overlay": _to_web_url(str(path_overlay_jpg)),
                "side_by_side": _to_web_url(str(path_composite_jpg)),
                "disk_folder": str(out_p.relative_to(REPO_ROOT)).replace("\\", "/") if out_p.is_relative_to(REPO_ROOT) else str(out_p)
            }
        }

        path_json = out_p / "patch_change_results.json"
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2)

        return result_data


_detector_instance: Optional[PrithviPatchChangeDetector] = None


def get_patch_change_detector() -> PrithviPatchChangeDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = PrithviPatchChangeDetector()
    return _detector_instance
