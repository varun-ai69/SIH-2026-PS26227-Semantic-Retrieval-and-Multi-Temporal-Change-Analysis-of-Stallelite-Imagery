"""
backend/services/clip_change_typer.py
======================================
RemoteCLIP Foundation-Model Vision-Language Change Classifier
=============================================================
Replaces manual threshold heuristics with zero-shot RemoteCLIP ViT-B-32
inference directly on candidate patch crops (Before T1 vs After T2).

Key Capabilities:
1. Zero-shot land-cover state classification for T1 and T2 crops using
   satellite-specific text prompts.
2. Change category prediction with softmax probability distribution.
3. Multimodal fusion: reinforces vision-language probability with physical
   multi-spectral deltas (ΔNDBI, ΔNDVI, ΔNDWI).
4. Spatial cluster consensus aggregation (majority vote, average confidence,
   ground area, and clean interpretation).
"""

from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch

from backend.services.encoder import get_encoder

logger = logging.getLogger("clip_change_typer")

# Standard Land-Cover State Prompts for Remote Sensing
LAND_COVER_PROMPTS = {
    "Vegetation": "a satellite image of green vegetation, trees, forest canopy, agricultural crops or farmland",
    "Built-Up": "a satellite image of concrete buildings, residential structures, urban roads, or settlement",
    "Bare Land": "a satellite image of exposed bare soil, cleared earth, open ground, or excavation",
    "Water": "a satellite image of a river, lake, canal, reservoir, or open water body",
    "Industrial Area": "a satellite image of industrial warehouses, commercial factories, or paved storage yards",
}

# Change Categories and Satellite Prompts
CHANGE_CATEGORY_PROMPTS = {
    "New Construction": "new building construction, concrete foundations, and urban structural development",
    "Land Clearance": "vegetation clearance, tree felling, bare ground excavation, and deforested terrain",
    "Road Development": "new road construction, ground grading, and asphalt transport paving",
    "Vegetation Regrowth": "vegetation regrowth, seasonal greening, and agricultural crop emergence",
    "Water Body Expansion": "water body expansion, rising water levels, reservoir filling, or flooding",
    "Water Body Recession": "water body drying, shrinking shoreline, or receding water level",
    "Ground Activity": "surface soil disturbance, earth moving, and site grading activity",
    "Stable Terrain": "stable terrain with no significant physical land-cover change",
}


class RemoteCLIPChangeTyper:
    """
    Evaluates Before (T1) and After (T2) patch crops with RemoteCLIP ViT-B-32
    to determine semantic states, change type probabilities, and cluster consensus.
    """

    def __init__(self):
        self.encoder = get_encoder()
        self.device = self.encoder.device
        self.model = self.encoder.model
        self.tokenizer = self.encoder.tokenizer
        self.preprocess = self.encoder.preprocess

        # Precompute and cache text prompt embeddings in memory
        self._init_prompt_embeddings()

    def _init_prompt_embeddings(self):
        """Precomputes normalized text embeddings for all state and change prompts."""
        logger.info("Precomputing RemoteCLIP text prompt embeddings...")
        
        # 1. State Prompts
        self.state_names = list(LAND_COVER_PROMPTS.keys())
        state_texts = [LAND_COVER_PROMPTS[k] for k in self.state_names]
        tokens = self.tokenizer(state_texts).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_text(tokens)
            emb /= emb.norm(dim=-1, keepdim=True)
            self.state_embeddings = emb  # [N_states, 512]

        # 2. Change Category Prompts
        self.change_names = list(CHANGE_CATEGORY_PROMPTS.keys())
        change_texts = [CHANGE_CATEGORY_PROMPTS[k] for k in self.change_names]
        tokens_c = self.tokenizer(change_texts).to(self.device)
        with torch.no_grad():
            emb_c = self.model.encode_text(tokens_c)
            emb_c /= emb_c.norm(dim=-1, keepdim=True)
            self.change_embeddings = emb_c  # [N_changes, 512]

        logger.info("RemoteCLIP text prompt embeddings ready.")

    def _to_pil(self, crop: Any) -> Image.Image:
        """Converts numpy array (3, H, W) or (H, W, 3) or PIL Image to RGB PIL Image."""
        if isinstance(crop, Image.Image):
            return crop.convert("RGB")

        if crop.ndim == 3 and crop.shape[0] == 3:
            arr = np.transpose(crop, (1, 2, 0))
        else:
            arr = crop

        if arr.dtype != np.uint8:
            if np.max(arr) <= 1.0:
                arr = (arr * 255.0).astype(np.uint8)
            else:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, mode="RGB")

    def classify_candidates(
        self,
        candidates: List[Dict[str, Any]],
        rgb_before: Any,
        rgb_after: Any,
        mb_before: Optional[np.ndarray] = None,
        mb_after: Optional[np.ndarray] = None,
        band_descriptions: Optional[List[str]] = None,
        band_descriptions_b: Optional[List[str]] = None,
        band_descriptions_a: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Runs RemoteCLIP vision-language inference on every candidate patch crop.
        Predicts:
        - state_t1 and state_t2 with confidence
        - predicted_change_type with probability distribution
        - multimodal fusion with spectral deltas (ΔNDBI, ΔNDVI, ΔNDWI)
        """
        if not candidates:
            return []

        # Prepare batch of T1 and T2 crops
        crops_t1, crops_t2 = [], []
        for cand in candidates:
            x0, y0, x1, y1 = cand["bbox"]
            if isinstance(rgb_before, Image.Image):
                crop_b = rgb_before.crop((x0, y0, x1, y1)).convert("RGB")
            else:
                crop_b = self._to_pil(rgb_before[y0:y1, x0:x1])

            if isinstance(rgb_after, Image.Image):
                crop_a = rgb_after.crop((x0, y0, x1, y1)).convert("RGB")
            else:
                crop_a = self._to_pil(rgb_after[y0:y1, x0:x1])

            crops_t1.append(crop_b)
            crops_t2.append(crop_a)

        # Encode T1 crops
        t1_tensors = torch.stack([self.preprocess(img) for img in crops_t1]).to(self.device)
        t2_tensors = torch.stack([self.preprocess(img) for img in crops_t2]).to(self.device)

        with torch.no_grad():
            feats_t1 = self.model.encode_image(t1_tensors)
            feats_t1 /= feats_t1.norm(dim=-1, keepdim=True)  # [B, 512]

            feats_t2 = self.model.encode_image(t2_tensors)
            feats_t2 /= feats_t2.norm(dim=-1, keepdim=True)  # [B, 512]

            # 1. State similarities & softmax probabilities
            # Temperature scaling: 100.0 (standard CLIP logit scale)
            logit_scale = getattr(self.model, "logit_scale", torch.tensor(4.6052)).exp()

            sims_t1 = (feats_t1 @ self.state_embeddings.T) * logit_scale
            probs_t1 = torch.softmax(sims_t1, dim=-1).cpu().numpy()  # [B, N_states]

            sims_t2 = (feats_t2 @ self.state_embeddings.T) * logit_scale
            probs_t2 = torch.softmax(sims_t2, dim=-1).cpu().numpy()  # [B, N_states]

            # 2. Difference vector Δv = (v_T2 - v_T1)
            diff_vec = feats_t2 - feats_t1
            diff_vec /= (diff_vec.norm(dim=-1, keepdim=True) + 1e-9)

            sims_diff = (diff_vec @ self.change_embeddings.T) * (logit_scale * 0.5)
            probs_diff = torch.softmax(sims_diff, dim=-1).cpu().numpy()  # [B, N_changes]

        from backend.services.semantic_change_classifier import SemanticChangeClassifier

        classified_candidates = []
        for i, cand in enumerate(candidates):
            c_dict = dict(cand)
            x0, y0, x1, y1 = cand["bbox"]

            # Vision-language model priors (RemoteCLIP zero-shot inference)
            clip_t1_idx = int(np.argmax(probs_t1[i]))
            clip_t2_idx = int(np.argmax(probs_t2[i]))
            clip_t1_name = self.state_names[clip_t1_idx]
            clip_t2_name = self.state_names[clip_t2_idx]
            t1_conf = round(float(probs_t1[i][clip_t1_idx]), 4)
            t2_conf = round(float(probs_t2[i][clip_t2_idx]), 4)

            # Change probabilities from difference projection
            change_probs = {
                name: round(float(probs_diff[i][k]), 4)
                for k, name in enumerate(self.change_names)
            }

            # Exact per-pixel band calculation across all 4,096 pixels of the patch crop
            if mb_before is not None and mb_after is not None:
                crop_b = mb_before[:, y0:y1, x0:x1]
                crop_a = mb_after[:, y0:y1, x0:x1]
                pixel_analysis = SemanticChangeClassifier.compute_patch_pixel_analysis(
                    crop_b=crop_b,
                    crop_a=crop_a,
                    band_descriptions=band_descriptions,
                    band_descriptions_b=band_descriptions_b,
                    band_descriptions_a=band_descriptions_a,
                    clip_hint_t1=clip_t1_name,
                    clip_hint_t2=clip_t2_name,
                )
                t1_name = pixel_analysis["dominant_t1_class"]
                t2_name = pixel_analysis["dominant_t2_class"]
                primary_type = pixel_analysis["predicted_type"]
                trans_label = pixel_analysis["transition_label"]
                ind_b = pixel_analysis["indices_t1"]
                ind_a = pixel_analysis["indices_t2"]
                deltas = pixel_analysis["deltas"]
                pixel_stats = pixel_analysis["pixel_stats"]
            else:
                # Fallback to scalar values if multi-band rasters unavailable
                ind_b = cand.get("indices_t1") or {}
                ind_a = cand.get("indices_t2") or {}
                ndvi_1, ndwi_1, ndbi_1 = ind_b.get("ndvi", 0.0), ind_b.get("ndwi", 0.0), ind_b.get("ndbi", 0.0)
                ndvi_2, ndwi_2, ndbi_2 = ind_a.get("ndvi", 0.0), ind_a.get("ndwi", 0.0), ind_a.get("ndbi", 0.0)
                deltas = cand.get("deltas") or {
                    "delta_ndvi": ndvi_2 - ndvi_1,
                    "delta_ndwi": ndwi_2 - ndwi_1,
                    "delta_ndbi": ndbi_2 - ndbi_1
                }
                def classify_state_by_values(ndvi_val: float, ndwi_val: float, ndbi_val: float, clip_hint: str) -> str:
                    if (ndwi_val > 0.0 and ndvi_val < 0.15) or ndwi_val >= 0.15:
                        return "Water Body"
                    if clip_hint in ("Built-Up / Urban", "Industrial Area") or (ndbi_val >= 0.05 and ndbi_val > ndvi_val):
                        return "Built-Up / Urban"
                    if ndvi_val >= 0.35 and ndbi_val < 0.05:
                        return "Dense Forest"
                    if 0.20 <= ndvi_val < 0.35 and ndvi_val > ndbi_val:
                        return "Farmland / Crops"
                    return "Bare Soil"

                t1_name = classify_state_by_values(ndvi_1, ndwi_1, ndbi_1, clip_t1_name)
                t2_name = classify_state_by_values(ndvi_2, ndwi_2, ndbi_2, clip_t2_name)
                d_ndbi = deltas.get("delta_ndbi", 0.0)
                d_ndvi = deltas.get("delta_ndvi", 0.0)
                d_ndwi = deltas.get("delta_ndwi", 0.0)
                primary_type, trans_label, _ = SemanticChangeClassifier.classify_by_dominant_delta(
                    delta_ndvi=d_ndvi,
                    delta_ndwi=d_ndwi,
                    delta_ndbi=d_ndbi,
                    dom_t1_name=t1_name,
                    dom_t2_name=t2_name,
                )
                pixel_stats = {}

            # Multimodal calibrated confidence score
            clip_conf = (t1_conf + t2_conf + change_probs.get(primary_type, 0.5)) / 3.0
            d_ndbi = deltas.get("delta_ndbi", 0.0)
            d_ndvi = deltas.get("delta_ndvi", 0.0)
            if primary_type == "New Construction" and d_ndbi > 0:
                clip_conf = min(1.0, clip_conf + 0.10)
            elif primary_type == "Land Clearance" and d_ndvi < 0:
                clip_conf = min(1.0, clip_conf + 0.10)
            elif primary_type == "Vegetation Regrowth" and d_ndvi > 0:
                clip_conf = min(1.0, clip_conf + 0.10)

            calibrated_confidence = round(float(np.clip(clip_conf, 0.50, 0.99)), 4)

            c_dict["indices_t1"] = ind_b
            c_dict["indices_t2"] = ind_a
            c_dict["deltas"] = deltas
            c_dict["predicted_type"] = primary_type
            c_dict["transition_label"] = trans_label
            c_dict["confidence"] = calibrated_confidence
            c_dict["t1_state"] = {"name": t1_name, "confidence": t1_conf}
            c_dict["t2_state"] = {"name": t2_name, "confidence": t2_conf}
            c_dict["pixel_stats"] = pixel_stats
            c_dict["change_probabilities"] = change_probs
            c_dict["ai_model"] = "Per-Pixel Bands (NDVI/NDWI/NDBI) + RemoteCLIP Disambiguation"

            classified_candidates.append(c_dict)

        return classified_candidates

    @classmethod
    def semantic_cluster_candidates(
        cls,
        classified_candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Clusters candidates semantically by their RemoteCLIP predicted change category
        and state transition (e.g. Land Clearance, Vegetation Regrowth), rather than
        purely spatial/geometric adjacency.
        Sorts clusters by patch count descending so Cluster #1 is the primary semantic driver.
        Assigns clean cluster_id (1, 2, 3...) to each candidate.
        """
        if not classified_candidates:
            return []

        # Group by RemoteCLIP semantic change type (e.g. Land Clearance, New Construction, Vegetation Regrowth)
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for c in classified_candidates:
            ptype = c.get("predicted_type", "Surface Activity")
            groups.setdefault(ptype, []).append(c)

        # Sort groups by size descending (largest semantic group gets Cluster #1)
        sorted_groups = sorted(groups.items(), key=lambda kv: len(kv[1]), reverse=True)

        clustered_candidates = []
        for cluster_idx, (ptype, patch_list) in enumerate(sorted_groups, start=1):
            for p in patch_list:
                p_copy = dict(p)
                p_copy["cluster_id"] = cluster_idx
                p_copy["cluster_size"] = len(patch_list)
                clustered_candidates.append(p_copy)

        return clustered_candidates

    @classmethod
    def aggregate_cluster_consensus(
        cls,
        classified_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Groups candidates by cluster_id and determines cluster-level consensus
        via RemoteCLIP probability voting and average confidence.
        """
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {}
        for cand in classified_candidates:
            cid = cand.get("cluster_id")
            if cid is not None:
                cluster_groups.setdefault(cid, []).append(cand)

        cluster_results = {}
        for cid, patches in cluster_groups.items():
            t1_states = [p["t1_state"]["name"] for p in patches if "t1_state" in p]
            t2_states = [p["t2_state"]["name"] for p in patches if "t2_state" in p]
            types = [p["predicted_type"] for p in patches if "predicted_type" in p]

            majority_t1 = Counter(t1_states).most_common(1)[0][0] if t1_states else "Unknown"
            majority_t2 = Counter(t2_states).most_common(1)[0][0] if t2_states else "Unknown"
            majority_type = Counter(types).most_common(1)[0][0] if types else "Surface Activity"

            mean_conf = round(float(np.mean([p.get("confidence", 0.85) for p in patches])), 4)
            area_m2 = len(patches) * 64 * 64 * 100

            # Spectral averages
            deltas = [p.get("deltas", {}) for p in patches]
            avg_d_ndvi = round(float(np.mean([d.get("delta_ndvi", 0.0) for d in deltas])), 4)
            avg_d_ndwi = round(float(np.mean([d.get("delta_ndwi", 0.0) for d in deltas])), 4)
            avg_d_ndbi = round(float(np.mean([d.get("delta_ndbi", 0.0) for d in deltas])), 4)

            from backend.services.semantic_change_classifier import SemanticChangeClassifier

            majority_type, transition_str, interpretation = SemanticChangeClassifier.classify_by_dominant_delta(
                delta_ndvi=avg_d_ndvi,
                delta_ndwi=avg_d_ndwi,
                delta_ndbi=avg_d_ndbi,
                dom_t1_name=majority_t1,
                dom_t2_name=majority_t2,
            )

            # Align majority_t2 with the physical transition
            if majority_type == "New Construction":
                majority_t2 = "Built-Up"
            elif majority_type in ("Vegetation Increase", "Vegetation Regrowth"):
                majority_t2 = "Vegetation"
            elif majority_type in ("Vegetation Decrease", "Land Clearance"):
                majority_t2 = "Bare Land" if majority_t1 == "Vegetation" else majority_t2
            elif majority_type == "Water Body Expansion":
                majority_t2 = "Water"
            elif majority_type == "Water Body Recession":
                majority_t2 = "Bare Land" if majority_t1 == "Water" else majority_t2

            # Propagate cluster majority consensus to ALL member patches in this cluster
            for p in patches:
                p["predicted_type"] = majority_type
                p["transition_label"] = transition_str
                p["cluster_majority_type"] = majority_type
                p["dominant_t1_class"] = majority_t1
                p["dominant_t2_class"] = majority_t2

            # Mean absolute indices for T1 and T2
            t1_ndvis = [p.get("indices_t1", {}).get("ndvi", 0.0) for p in patches if "indices_t1" in p]
            t2_ndvis = [p.get("indices_t2", {}).get("ndvi", 0.0) for p in patches if "indices_t2" in p]
            t1_ndbis = [p.get("indices_t1", {}).get("ndbi", 0.0) for p in patches if "indices_t1" in p]
            t2_ndbis = [p.get("indices_t2", {}).get("ndbi", 0.0) for p in patches if "indices_t2" in p]
            t1_ndwis = [p.get("indices_t1", {}).get("ndwi", 0.0) for p in patches if "indices_t1" in p]
            t2_ndwis = [p.get("indices_t2", {}).get("ndwi", 0.0) for p in patches if "indices_t2" in p]

            mean_t1_ndvi = round(float(np.mean(t1_ndvis)), 3) if t1_ndvis else 0.0
            mean_t2_ndvi = round(float(np.mean(t2_ndvis)), 3) if t2_ndvis else 0.0
            mean_t1_ndbi = round(float(np.mean(t1_ndbis)), 3) if t1_ndbis else 0.0
            mean_t2_ndbi = round(float(np.mean(t2_ndbis)), 3) if t2_ndbis else 0.0
            mean_t1_ndwi = round(float(np.mean(t1_ndwis)), 3) if t1_ndwis else 0.0
            mean_t2_ndwi = round(float(np.mean(t2_ndwis)), 3) if t2_ndwis else 0.0

            cluster_results[str(cid)] = {
                "cluster_id": cid,
                "num_patches": len(patches),
                "area_m2": area_m2,
                "predicted_type": majority_type,
                "transition_label": majority_type,
                "cluster_transition": transition_str,
                "interpretation": interpretation,
                "confidence": mean_conf,
                "confidence_pct": round(mean_conf * 100, 1),
                "dominant_t1_class": majority_t1,
                "dominant_t2_class": majority_t2,
                "majority_class_t1": majority_t1,
                "majority_class_t2": majority_t2,
                "mean_t1_ndvi": mean_t1_ndvi,
                "mean_t2_ndvi": mean_t2_ndvi,
                "mean_t1_ndbi": mean_t1_ndbi,
                "mean_t2_ndbi": mean_t2_ndbi,
                "mean_t1_ndwi": mean_t1_ndwi,
                "mean_t2_ndwi": mean_t2_ndwi,
                "mean_delta_ndvi": avg_d_ndvi,
                "mean_delta_ndwi": avg_d_ndwi,
                "mean_delta_ndbi": avg_d_ndbi,
                "ai_model": "Per-Pixel Bands (NDVI/NDWI/NDBI) + RemoteCLIP Disambiguation",
                "patches": [
                    {
                        "patch_id": p.get("patch_id"),
                        "row": p.get("row"),
                        "col": p.get("col"),
                        "type": p.get("predicted_type"),
                        "confidence": p.get("confidence"),
                        "transition": p.get("transition_label"),
                        "t1": p.get("t1_state", {}).get("name"),
                        "t2": p.get("t2_state", {}).get("name"),
                    }
                    for p in patches
                ]
            }

        return cluster_results


# Shared singleton instance
_global_typer: Optional[RemoteCLIPChangeTyper] = None


def get_change_typer() -> RemoteCLIPChangeTyper:
    global _global_typer
    if _global_typer is None:
        _global_typer = RemoteCLIPChangeTyper()
    return _global_typer
