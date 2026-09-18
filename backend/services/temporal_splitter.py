"""
backend/services/temporal_splitter.py
======================================
Prithvi Multi-Temporal Splitting Engine
======================================
Evaluates multi-year temporal observations for a given geographic site footprint.
Using NASA-IBM Prithvi-EO-2.0-300M multi-spectral foundation model:
  1. Computes latent embeddings across chronological epochs.
  2. Evaluates between-segment to within-segment variance across candidate split dates:
        Score(s) = ||mu_{<s} - mu_{>=s}||^2 / (var_{within} + eps)
  3. Identifies the optimal change onset split s*.
  4. Returns strictly the 2 milestone observations:
        T_before = observation[s* - 1] (last clean baseline before change)
        T_after  = observation[s*]     (first emergence/onset of change)
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

import torch
from backend.services.prithvi_encoder import get_prithvi_encoder

logger = logging.getLogger("temporal_splitter")


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def resolve_file_path(path_str: Optional[str]) -> Optional[Path]:
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


class PrithviTemporalSplitter:
    """
    Evaluates a time series of >= 5 clean satellite observations for a geographic site,
    running Prithvi-EO-2.0-300M temporal step detection to identify the exact change
    transition point and return the milestone pair (T_before, T_after).
    """

    def __init__(self):
        self.encoder = get_prithvi_encoder()

    def _load_multiband(self, file_path: str) -> Optional[np.ndarray]:
        """Loads multi-band array (C, H, W) from GeoTIFF."""
        resolved = resolve_file_path(file_path)
        if not resolved or not resolved.is_file():
            return None
        if rasterio is None:
            return None
        try:
            with rasterio.open(str(resolved)) as ds:
                arr = ds.read().astype(np.float32)
                return arr
        except Exception as e:
            logger.warning(f"Failed to read raster {file_path}: {e}")
            return None

    def evaluate_series(
        self,
        clean_records: List[Dict[str, Any]],
        grid_patches: int = 4  # 4x4 = 16 spatial patches for local change sensitivity
    ) -> Dict[str, Any]:
        """
        Takes chronologically sorted clean observation records (length >= 2).
        If length < 5: Returns baseline (first) and subsequent (last) without heavy step splitting.
        If length >= 5: Performs Prithvi temporal step detection across all candidate split points.

        Returns:
            {
                "method": "prithvi_step_detection" | "direct_pair",
                "total_clean_epochs": int,
                "split_index": int,
                "split_date_before": str,
                "split_date_after": str,
                "onset_bracket": str,
                "step_score": float,
                "all_split_scores": List[Dict[str, Any]],
                "selected_before_record": Dict,
                "selected_after_record": Dict
            }
        """
        n_epochs = len(clean_records)
        if n_epochs < 2:
            raise ValueError(f"Need at least 2 clean observations for temporal comparison, got {n_epochs}")

        # Case 1: Less than 5 observations -> Fallback to earliest and latest
        if n_epochs < 5:
            rec_b = clean_records[0]
            rec_a = clean_records[-1]
            date_b = str(rec_b["acquisition_date"])[:10]
            date_a = str(rec_a["acquisition_date"])[:10]
            return {
                "method": "direct_baseline_latest",
                "total_clean_epochs": n_epochs,
                "split_index": n_epochs - 1,
                "split_date_before": date_b,
                "split_date_after": date_a,
                "onset_bracket": f"{date_b} → {date_a}",
                "step_score": 1.0,
                "all_split_scores": [],
                "selected_before_record": rec_b,
                "selected_after_record": rec_a
            }

        # Case 2: >= 5 observations -> Run Prithvi Multi-Spectral Step Detection
        logger.info(f"Running Prithvi-EO-2.0-300M Temporal Splitting Engine on {n_epochs} clean epochs...")

        # Load rasters for all clean epochs
        epoch_rasters = []
        valid_records = []
        for rec in clean_records:
            arr = self._load_multiband(rec.get("file_path"))
            if arr is not None and arr.ndim >= 3:
                epoch_rasters.append(arr)
                valid_records.append(rec)

        if len(epoch_rasters) < 2:
            # Fallback if raster loading failed
            return {
                "method": "fallback_metadata",
                "total_clean_epochs": n_epochs,
                "split_index": 1,
                "split_date_before": str(clean_records[0]["acquisition_date"])[:10],
                "split_date_after": str(clean_records[-1]["acquisition_date"])[:10],
                "onset_bracket": f"{str(clean_records[0]['acquisition_date'])[:10]} → {str(clean_records[-1]['acquisition_date'])[:10]}",
                "step_score": 0.0,
                "all_split_scores": [],
                "selected_before_record": clean_records[0],
                "selected_after_record": clean_records[-1]
            }

        K = len(epoch_rasters)
        C, H, W = epoch_rasters[0].shape
        pw = W // grid_patches
        ph = H // grid_patches

        # Extract patch crops for all epochs
        all_epoch_patches = []
        for arr in epoch_rasters:
            patches = []
            for r in range(grid_patches):
                for c in range(grid_patches):
                    crop = arr[:, r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
                    patches.append(crop)
            all_epoch_patches.append(patches)

        # Compute Prithvi 1024-D embeddings for each epoch
        epoch_embeddings = []
        for idx, patches in enumerate(all_epoch_patches):
            vecs = self.encoder.encode_multiband_patches(patches, batch_size=16)
            epoch_embeddings.append(vecs)

        # Array of shape (K, num_patches, 1024)
        F = np.stack(epoch_embeddings, axis=0)
        num_patches = F.shape[1]

        # Evaluate Step Score for every candidate split s in [1, K-1]
        split_evaluations = []
        best_s = 1
        max_total_score = -1.0

        for s in range(1, K):
            date_pre = str(valid_records[s - 1]["acquisition_date"])[:10]
            date_post = str(valid_records[s]["acquisition_date"])[:10]

            F_pre = F[:s]    # (s, P, 1024)
            F_post = F[s:]   # (K - s, P, 1024)

            # Mean embedding per patch across segments
            mu_pre = F_pre.mean(axis=0)    # (P, 1024)
            mu_post = F_post.mean(axis=0)  # (P, 1024)

            # Contrast (between-segment squared distance) per patch
            delta_mu = mu_post - mu_pre
            between_var = np.sum(delta_mu ** 2, axis=-1)  # (P,)

            # Within-segment variance per patch (captures seasonal noise)
            var_pre = np.mean(np.sum((F_pre - mu_pre[np.newaxis, :, :]) ** 2, axis=-1), axis=0) if s > 1 else np.zeros(num_patches)
            var_post = np.mean(np.sum((F_post - mu_post[np.newaxis, :, :]) ** 2, axis=-1), axis=0) if (K - s) > 1 else np.zeros(num_patches)
            within_var = (s * var_pre + (K - s) * var_post) / K + 1e-4

            # Patch-level step score (F-ratio style)
            patch_scores = between_var / within_var

            # Aggregate score: use 90th percentile to detect localized changes strongly
            step_score_p90 = float(np.percentile(patch_scores, 90))
            step_score_mean = float(np.mean(patch_scores))

            # Composite metric prioritizing strong localized onset
            combined_metric = step_score_p90 * 0.7 + step_score_mean * 0.3

            split_evaluations.append({
                "split_index": s,
                "date_before": date_pre,
                "date_after": date_post,
                "onset_bracket": f"{date_pre} → {date_post}",
                "score_p90": round(step_score_p90, 4),
                "score_mean": round(step_score_mean, 4),
                "combined_metric": round(combined_metric, 4)
            })

            if combined_metric > max_total_score:
                max_total_score = combined_metric
                best_s = s

        selected_b = valid_records[best_s - 1]
        selected_a = valid_records[best_s]
        best_date_b = str(selected_b["acquisition_date"])[:10]
        best_date_a = str(selected_a["acquisition_date"])[:10]

        logger.info(
            f"Prithvi Splitting Engine selected optimal split s*={best_s} "
            f"({best_date_b} → {best_date_a}) with step score {max_total_score:.4f}"
        )

        device_str = str(self.encoder.device)
        device_label = "CPU"
        if "cuda" in device_str and torch.cuda.is_available():
            try:
                device_label = torch.cuda.get_device_name(self.encoder.device)
            except Exception:
                device_label = "NVIDIA CUDA GPU"

        return {
            "method": "prithvi_multi_spectral_step_detection",
            "device": device_str,
            "device_name": device_label,
            "total_clean_epochs": K,
            "split_index": best_s,
            "split_date_before": best_date_b,
            "split_date_after": best_date_a,
            "onset_bracket": f"{best_date_b} → {best_date_a}",
            "step_score": round(max_total_score, 4),
            "all_split_scores": split_evaluations,
            "selected_before_record": selected_b,
            "selected_after_record": selected_a
        }


# Global singleton instance
_splitter_instance: Optional[PrithviTemporalSplitter] = None


def get_temporal_splitter() -> PrithviTemporalSplitter:
    global _splitter_instance
    if _splitter_instance is None:
        _splitter_instance = PrithviTemporalSplitter()
    return _splitter_instance
