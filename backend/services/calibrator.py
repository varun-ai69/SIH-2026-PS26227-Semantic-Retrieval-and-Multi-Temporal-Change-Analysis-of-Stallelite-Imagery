"""
backend/services/calibrator.py
==============================
Training-Free Retrieval Score Calibration & Hubness Suppression
PS Sections: 2.2.1, 2.2.4 (Semantic & Multimodal Retrieval Calibration)

Implements:
1. QB-Norm (Querybank Normalization) with Dynamic Inverted Softmax (DIS).
   Reference: Bogolin et al., "Cross Modal Retrieval with Querybank Normalisation", CVPR 2022.
2. Exact 512 RSICD Train Captions Query Bank (as used in NetSight and published literature).
3. Hubness detection and Dynamic Gating:
   If the top-1 result is not a hub, raw scores are preserved.
   If the top-1 result is a hub, Inverted Softmax dampens the hub and promotes specific matches.
"""

from __future__ import annotations

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

logger = logging.getLogger("Calibrator")

DEFAULT_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "calibration"
QUERY_BANK_VEC_FILE = DEFAULT_CACHE_DIR / "query_bank_remoteclip.npy"
QUERY_BANK_TXT_FILE = DEFAULT_CACHE_DIR / "rsicd_512_captions.json"


class RetrievalCalibrator:
    """
    Manages QB-Norm Dynamic Inverted Softmax calibration using the 512 RSICD Query Bank.
    """

    def __init__(
        self,
        beta: float = 20.0,
        hub_topk: int = 5,
        matrix_path: Optional[Path] = None,
        captions_path: Optional[Path] = None
    ):
        self.beta = beta
        self.hub_topk = hub_topk
        self.matrix_path = Path(matrix_path) if matrix_path else QUERY_BANK_VEC_FILE
        self.captions_path = Path(captions_path) if captions_path else QUERY_BANK_TXT_FILE
        self._bank_embeddings: Optional[np.ndarray] = None  # Shape: (512, 512)
        self._captions: List[str] = []
        self._load_bank()

    def _load_bank(self) -> None:
        """Loads cached 512 RSICD query bank embeddings."""
        if self.matrix_path.exists():
            try:
                self._bank_embeddings = np.load(str(self.matrix_path)).astype(np.float32)
                # Ensure L2 normalized
                self._bank_embeddings /= (np.linalg.norm(self._bank_embeddings, axis=1, keepdims=True) + 1e-9)
                logger.info(f"Loaded {self._bank_embeddings.shape[0]} RSICD query bank embeddings from {self.matrix_path}")
            except Exception as e:
                logger.error(f"Error loading {self.matrix_path}: {e}")

        if self.captions_path.exists():
            try:
                with open(self.captions_path, "r", encoding="utf-8") as f:
                    self._captions = json.load(f)
            except Exception:
                pass

    @property
    def is_ready(self) -> bool:
        return self._bank_embeddings is not None and len(self._bank_embeddings) > 0

    def calibrate_candidates(
        self,
        query_vec: np.ndarray,
        candidate_ids: List[str],
        candidate_vecs: np.ndarray,
        raw_scores: np.ndarray,
        beta: Optional[float] = None,
        mode: str = "auto"
    ) -> Tuple[List[str], np.ndarray, np.ndarray, List[bool]]:
        """
        Applies QB-Norm Dynamic Inverted Softmax (DIS) calibration to candidate tiles.

        Args:
            query_vec: L2-normalized query embedding (512,)
            candidate_ids: List of tile_ids (K,)
            candidate_vecs: Matrix of candidate tile embeddings (K, 512)
            raw_scores: Array of raw cosine similarities (K,)
            beta: Inverse temperature parameter (default 20.0)
            mode: "auto" (dynamic gating: only calibrate if top-1 is a hub),
                  "always" (unconditionally calibrate all candidates),
                  "off" (return raw scores).

        Returns:
            ranked_ids: Sorted list of tile_ids
            calibrated_scores: Calibrated scores matching ranked_ids
            raw_scores_sorted: Raw scores corresponding to ranked_ids
            is_hub_flags: Boolean flags indicating which candidates were hubs
        """
        if not self.is_ready or mode == "off" or len(candidate_ids) == 0:
            order = np.argsort(-raw_scores)
            return (
                [candidate_ids[i] for i in order],
                raw_scores[order],
                raw_scores[order],
                [False] * len(candidate_ids)
            )

        b = float(beta or self.beta)
        B = self._bank_embeddings  # Shape: (512, 512)
        X = candidate_vecs.astype(np.float32)  # Shape: (K, 512)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

        # S_b shape: (512, K) -> Cosine similarity of every bank query to every candidate tile
        S_b = np.dot(B, X.T)

        # Hub detection:
        # A tile is flagged as a hub if it appears in top_k for multiple unrelated bank queries,
        # or if its mean similarity across bank queries is significantly elevated.
        M, K = S_b.shape
        hub_rank_k = min(self.hub_topk, K)
        top_k_indices = np.argsort(-S_b, axis=1)[:, :hub_rank_k]  # (512, hub_topk)

        hub_counts = np.zeros(K, dtype=int)
        for row in top_k_indices:
            for idx in row:
                hub_counts[idx] += 1

        mean_bank_sim = np.mean(S_b, axis=0)  # (K,)
        global_mean = float(np.mean(mean_bank_sim))
        global_std = float(np.std(mean_bank_sim)) + 1e-6
        # Flag tile as hub if it wins top-k in >= 5 bank queries or is > 1.25 std above bank mean
        is_hub_arr = (hub_counts >= 5) | (mean_bank_sim > (global_mean + 1.25 * global_std))

        # Dynamic Gating (from Bogolin et al. CVPR 2022):
        # If the raw top-1 result is NOT a hub, the query is clean; preserve raw scores.
        raw_top1_idx = int(np.argmax(raw_scores))
        if mode == "auto" and not is_hub_arr[raw_top1_idx]:
            order = np.argsort(-raw_scores)
            return (
                [candidate_ids[i] for i in order],
                raw_scores[order],
                raw_scores[order],
                [bool(is_hub_arr[i]) for i in order]
            )

        # Inverted Softmax:
        # Z_j = (1/beta) * (logsumexp(beta * S_b[:, j]) - log(M))
        beta_Sb = b * S_b  # (512, K)
        max_val = np.max(beta_Sb, axis=0, keepdims=True)
        logsumexp = max_val + np.log(np.sum(np.exp(beta_Sb - max_val), axis=0, keepdims=True) + 1e-9)
        z = (logsumexp[0] - np.log(M)) / b  # (K,)

        # Relative hub penalty: center z around its median so neutral tiles suffer 0 penalty
        z_baseline = float(np.median(z))
        penalty = np.maximum(0.0, z - z_baseline)

        # Penalize raw cosine scores
        calibrated = raw_scores - 0.75 * penalty
        calibrated = np.clip(calibrated, 0.0, 1.0)

        order = np.argsort(-calibrated)
        return (
            [candidate_ids[i] for i in order],
            calibrated[order],
            raw_scores[order],
            [bool(is_hub_arr[i]) for i in order]
        )


_calibrator_instance: Optional[RetrievalCalibrator] = None


def get_calibrator() -> RetrievalCalibrator:
    global _calibrator_instance
    if _calibrator_instance is None:
        _calibrator_instance = RetrievalCalibrator()
    return _calibrator_instance
