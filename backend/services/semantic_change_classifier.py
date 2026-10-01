"""
backend/services/semantic_change_classifier.py
==============================================
Multi-Spectral Semantic Change Classification Engine
=====================================================
Evaluates NDVI, NDWI, NDBI, and BSI across candidate change patches to identify
clean physical land cover states and cluster-level transition consensus.

Land Cover Hierarchy (Sentinel-2):
1. Water Body:        NDWI >= 0.20 and NDVI < 0.20
2. Dense Vegetation:   NDVI >= 0.40 and NDWI < 0.20
3. Sparse Vegetation:  0.20 <= NDVI < 0.40 and NDWI < 0.20
4. Built-Up Area:      NDVI < 0.25 and (NDBI >= 0.10 or NDBI >= BSI)
5. Bare Soil:          NDVI < 0.25 and (BSI > NDBI or NDBI < 0.10)
6. Mixed Land Cover:   Conflicting / transition pixels

Spectral Stability Filter:
Drops candidate patches ONLY if |ΔNDVI| < 0.002, |ΔNDWI| < 0.002, and |ΔNDBI| < 0.002
(strictly rejects identical patches while preserving real physical shifts).
"""

from collections import Counter
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("semantic_change_classifier")

# Clean, professional class definitions
LAND_COVER_CLASSES = {
    1: {"id": 1, "name": "Water Body", "color": "#0284c7", "desc": "Open water / reservoir / river"},
    2: {"id": 2, "name": "Dense Vegetation", "color": "#16a34a", "desc": "Forest / dense crops / canopy"},
    3: {"id": 3, "name": "Sparse Vegetation", "color": "#84cc16", "desc": "Grassland / shrubs / sparse crops"},
    4: {"id": 4, "name": "Built-Up Area", "color": "#ef4444", "desc": "Buildings / concrete / urban surfaces"},
    5: {"id": 5, "name": "Bare Soil", "color": "#d97706", "desc": "Exposed soil / sand / barren land"},
    6: {"id": 6, "name": "Mixed Land Cover", "color": "#a855f7", "desc": "Transition / mixed surfaces"},
}


class SemanticChangeClassifier:
    """
    Evaluates multi-spectral patch crops to classify before/after land cover states
    and cluster-level transitions.
    """

    @staticmethod
    def extract_bands(
        patch_crop: np.ndarray,
        band_descriptions: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Maps multi-band raster crop (C, H, W) to required spectral channels:
        Red, Green, Blue, NIR, SWIR.
        Handles Sentinel-2 9-band L2A (B01..B12), 5-band (Blue, Green, Red, NIR, SWIR),
        and 4-band (Red, Green, Blue, NIR) configurations flawlessly.
        """
        c, h, w = patch_crop.shape
        band_map = {}

        # 1. Match from explicit descriptions if description count matches channel count
        if band_descriptions and len(band_descriptions) == c:
            desc_lower = [str(d).lower().strip() for d in band_descriptions]
            for idx, name in enumerate(desc_lower):
                if name in ("red", "b04", "b4"):
                    band_map["red"] = patch_crop[idx].astype(np.float32)
                elif name in ("green", "b03", "b3"):
                    band_map["green"] = patch_crop[idx].astype(np.float32)
                elif name in ("blue", "b02", "b2"):
                    band_map["blue"] = patch_crop[idx].astype(np.float32)
                elif name in ("nir", "b08", "b8", "b8a"):
                    if "nir" not in band_map:
                        band_map["nir"] = patch_crop[idx].astype(np.float32)
                elif name in ("swir", "b11", "b12", "swir1", "swir2"):
                    if "swir" not in band_map:
                        band_map["swir"] = patch_crop[idx].astype(np.float32)

        # 2. Canonical channel index mapping based on exact channel count (C)
        if c >= 9:
            # Full Sentinel-2 L2A: 0:B01, 1:B02(Blue), 2:B03(Green), 3:B04(Red), 4:B05, 5:B08(NIR), 6:B8A, 7:B11(SWIR)
            band_map.setdefault("blue", patch_crop[1].astype(np.float32))
            band_map.setdefault("green", patch_crop[2].astype(np.float32))
            band_map.setdefault("red", patch_crop[3].astype(np.float32))
            band_map.setdefault("nir", patch_crop[5].astype(np.float32))
            band_map.setdefault("swir", patch_crop[7].astype(np.float32))
        elif c == 5:
            # 5-Band Standard: 0:Blue, 1:Green, 2:Red, 3:NIR, 4:SWIR
            band_map.setdefault("blue", patch_crop[0].astype(np.float32))
            band_map.setdefault("green", patch_crop[1].astype(np.float32))
            band_map.setdefault("red", patch_crop[2].astype(np.float32))
            band_map.setdefault("nir", patch_crop[3].astype(np.float32))
            band_map.setdefault("swir", patch_crop[4].astype(np.float32))
        elif c == 6:
            # 6-Band: 0:Blue, 1:Green, 2:Red, 3:RedEdge, 4:NIR, 5:SWIR
            band_map.setdefault("blue", patch_crop[0].astype(np.float32))
            band_map.setdefault("green", patch_crop[1].astype(np.float32))
            band_map.setdefault("red", patch_crop[2].astype(np.float32))
            band_map.setdefault("nir", patch_crop[4].astype(np.float32))
            band_map.setdefault("swir", patch_crop[5].astype(np.float32))
        elif c == 4:
            # 4-Band RGBN: 0:Red, 1:Green, 2:Blue, 3:NIR
            band_map.setdefault("red", patch_crop[0].astype(np.float32))
            band_map.setdefault("green", patch_crop[1].astype(np.float32))
            band_map.setdefault("blue", patch_crop[2].astype(np.float32))
            band_map.setdefault("nir", patch_crop[3].astype(np.float32))
            band_map.setdefault("swir", (0.8 * patch_crop[3] + 0.2 * patch_crop[0]).astype(np.float32))
        elif c == 3:
            # 3-Band True Color: 0:Red, 1:Green, 2:Blue
            band_map.setdefault("red", patch_crop[0].astype(np.float32))
            band_map.setdefault("green", patch_crop[1].astype(np.float32))
            band_map.setdefault("blue", patch_crop[2].astype(np.float32))
            band_map.setdefault("nir", patch_crop[1].astype(np.float32))
            band_map.setdefault("swir", patch_crop[0].astype(np.float32))

        return band_map

    @classmethod
    def compute_patch_indices(
        cls,
        patch_crop: np.ndarray,
        band_descriptions: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Computes mean NDVI, NDWI, NDBI, and BSI for a patch crop.
        """
        bands = cls.extract_bands(patch_crop, band_descriptions)
        blue = bands.get("blue")
        green = bands.get("green")
        red = bands.get("red")
        nir = bands.get("nir")
        swir = bands.get("swir")

        valid = np.ones(patch_crop.shape[1:], dtype=bool)
        if red is not None:
            valid &= (red > 0) & (~np.isnan(red)) & (~np.isinf(red))

        eps = 1e-6

        # NDVI = (NIR - Red) / (NIR + Red)
        ndvi_val = 0.0
        if nir is not None and red is not None:
            denom = nir + red
            denom[denom == 0] = eps
            arr = (nir - red) / denom
            v = arr[valid]
            if len(v) > 0:
                ndvi_val = float(np.clip(np.mean(v), -1.0, 1.0))

        # NDWI = (Green - NIR) / (Green + NIR)
        ndwi_val = 0.0
        if green is not None and nir is not None:
            denom = green + nir
            denom[denom == 0] = eps
            arr = (green - nir) / denom
            v = arr[valid]
            if len(v) > 0:
                ndwi_val = float(np.clip(np.mean(v), -1.0, 1.0))

        # NDBI = (SWIR - NIR) / (SWIR + NIR)
        ndbi_val = 0.0
        if swir is not None and nir is not None:
            denom = swir + nir
            denom[denom == 0] = eps
            arr = (swir - nir) / denom
            v = arr[valid]
            if len(v) > 0:
                ndbi_val = float(np.clip(np.mean(v), -1.0, 1.0))

        # BSI = ((SWIR + Red) - (NIR + Blue)) / ((SWIR + Red) + (NIR + Blue))
        bsi_val = 0.0
        if swir is not None and red is not None and nir is not None and blue is not None:
            numer = (swir + red) - (nir + blue)
            denom = (swir + red) + (nir + blue)
            denom[denom == 0] = eps
            arr = numer / denom
            v = arr[valid]
            if len(v) > 0:
                bsi_val = float(np.clip(np.mean(v), -1.0, 1.0))

        return {
            "ndvi": round(ndvi_val, 4),
            "ndwi": round(ndwi_val, 4),
            "ndbi": round(ndbi_val, 4),
            "bsi": round(bsi_val, 4),
        }

    @classmethod
    def compute_patch_pixel_analysis(
        cls,
        crop_b: np.ndarray,
        crop_a: np.ndarray,
        band_descriptions: Optional[List[str]] = None,
        band_descriptions_b: Optional[List[str]] = None,
        band_descriptions_a: Optional[List[str]] = None,
        clip_hint_t1: Optional[str] = None,
        clip_hint_t2: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Computes per-pixel NDVI, NDWI, NDBI for the 64x64 patch crop using actual band values.
        Classifies every individual pixel at T1 and T2 using physical multispectral indices.
        Uses RemoteCLIP encoder ONLY to disambiguate between Bare Land and Built-Up!
        Aggregates pixel-level transition counts to determine the patch-level change.
        """
        desc_b = band_descriptions_b if (band_descriptions_b and len(band_descriptions_b) == crop_b.shape[0]) else (band_descriptions if (band_descriptions and len(band_descriptions) == crop_b.shape[0]) else None)
        desc_a = band_descriptions_a if (band_descriptions_a and len(band_descriptions_a) == crop_a.shape[0]) else (band_descriptions if (band_descriptions and len(band_descriptions) == crop_a.shape[0]) else None)

        bands_b = cls.extract_bands(crop_b, desc_b)
        bands_a = cls.extract_bands(crop_a, desc_a)

        red_b, green_b, nir_b = bands_b.get("red"), bands_b.get("green"), bands_b.get("nir")
        swir_b = bands_b.get("swir")
        red_a, green_a, nir_a = bands_a.get("red"), bands_a.get("green"), bands_a.get("nir")
        swir_a = bands_a.get("swir")

        h, w = crop_b.shape[1], crop_b.shape[2]
        total_px = h * w
        eps = 1e-6

        # SWIR fallback if missing: 0.8 * NIR + 0.2 * Red
        if swir_b is None and nir_b is not None and red_b is not None:
            swir_b = 0.8 * nir_b + 0.2 * red_b
        if swir_a is None and nir_a is not None and red_a is not None:
            swir_a = 0.8 * nir_a + 0.2 * red_a

        # 1. Per-pixel indices for T1
        if nir_b is not None and red_b is not None:
            d_denom = nir_b + red_b
            d_denom[d_denom == 0] = eps
            ndvi_1 = np.clip((nir_b - red_b) / d_denom, -1.0, 1.0)
        else:
            ndvi_1 = np.zeros((h, w), dtype=np.float32)

        if green_b is not None and nir_b is not None:
            d_denom = green_b + nir_b
            d_denom[d_denom == 0] = eps
            ndwi_1 = np.clip((green_b - nir_b) / d_denom, -1.0, 1.0)
        else:
            ndwi_1 = np.zeros((h, w), dtype=np.float32)

        if swir_b is not None and nir_b is not None:
            d_denom = swir_b + nir_b
            d_denom[d_denom == 0] = eps
            ndbi_1 = np.clip((swir_b - nir_b) / d_denom, -1.0, 1.0)
        else:
            ndbi_1 = np.zeros((h, w), dtype=np.float32)

        # 2. Per-pixel indices for T2
        if nir_a is not None and red_a is not None:
            d_denom = nir_a + red_a
            d_denom[d_denom == 0] = eps
            ndvi_2 = np.clip((nir_a - red_a) / d_denom, -1.0, 1.0)
        else:
            ndvi_2 = np.zeros((h, w), dtype=np.float32)

        if green_a is not None and nir_a is not None:
            d_denom = green_a + nir_a
            d_denom[d_denom == 0] = eps
            ndwi_2 = np.clip((green_a - nir_a) / d_denom, -1.0, 1.0)
        else:
            ndwi_2 = np.zeros((h, w), dtype=np.float32)

        if swir_a is not None and nir_a is not None:
            d_denom = swir_a + nir_a
            d_denom[d_denom == 0] = eps
            ndbi_2 = np.clip((swir_a - nir_a) / d_denom, -1.0, 1.0)
        else:
            ndbi_2 = np.zeros((h, w), dtype=np.float32)

        d_ndvi = ndvi_2 - ndvi_1
        d_ndwi = ndwi_2 - ndwi_1
        d_ndbi = ndbi_2 - ndbi_1

        # 3. Classify each pixel:
        # 1: Water Body, 2: Dense Forest, 3: Farmland / Crops, 4: Built-Up / Urban, 5: Bare Soil
        def classify_grid(ndvi_grid, ndwi_grid, ndbi_grid, clip_hint):
            cls_grid = np.zeros((h, w), dtype=int)
            # Water Body: NDWI >= 0.15 or (NDWI > 0.0 and NDVI < 0.15)
            water_mask = (ndwi_grid >= 0.15) | ((ndwi_grid > 0.0) & (ndvi_grid < 0.15))
            cls_grid[water_mask] = 1

            # Dense Forest: NDVI >= 0.35 and not water
            forest_mask = (ndvi_grid >= 0.35) & (cls_grid == 0)
            cls_grid[forest_mask] = 2

            # Farmland / Crops: 0.20 <= NDVI < 0.35 and NDVI > NDBI and not water
            crops_mask = (ndvi_grid >= 0.20) & (ndvi_grid < 0.35) & (ndvi_grid > ndbi_grid) & (cls_grid == 0)
            cls_grid[crops_mask] = 3

            # Non-vegetated dry pixels (ndvi < 0.20): Bare Land vs Built-Up
            # Disambiguated by physical NDBI band values and RemoteCLIP encoder
            dry_mask = (cls_grid == 0)
            is_builtup_hint = False
            if np.any(dry_mask):
                mean_ndbi = float(np.mean(ndbi_grid[dry_mask]))
                mean_ndvi = float(np.mean(ndvi_grid[dry_mask]))
                # Physical spectral values: high NDBI (> 0.10 or NDBI > NDVI + 0.05) is built-up/concrete/roof
                if mean_ndbi >= 0.10 and mean_ndbi > mean_ndvi:
                    is_builtup_hint = True
                elif clip_hint in ("Built-Up", "Built-Up / Urban", "Industrial Area"):
                    is_builtup_hint = True
                elif clip_hint in ("Bare Land", "Bare Soil", "Vegetation", "Dense Forest", "Farmland / Crops"):
                    is_builtup_hint = False
                else:
                    is_builtup_hint = (mean_ndbi >= 0.05 and mean_ndbi > mean_ndvi)

            if is_builtup_hint:
                cls_grid[dry_mask] = 4  # Built-Up
            else:
                cls_grid[dry_mask] = 5  # Bare Land

            return cls_grid

        grid_1 = classify_grid(ndvi_1, ndwi_1, ndbi_1, clip_hint_t1)
        grid_2 = classify_grid(ndvi_2, ndwi_2, ndbi_2, clip_hint_t2)

        CLASS_MAP = {
            1: "Water",
            2: "Vegetation",
            3: "Vegetation",
            4: "Built-Up",
            5: "Bare Land"
        }

        # Dominant land cover across the 4,096 pixels
        counts_1 = np.bincount(grid_1.flatten(), minlength=6)
        counts_2 = np.bincount(grid_2.flatten(), minlength=6)

        dom_1_id = int(np.argmax(counts_1[1:])) + 1
        dom_2_id = int(np.argmax(counts_2[1:])) + 1
        dom_t1_name = CLASS_MAP.get(dom_1_id, "Bare Land")
        dom_t2_name = CLASS_MAP.get(dom_2_id, "Bare Land")

        # 4. Pixel-level transition counts
        clearing_px = int(np.sum(((grid_1 == 2) | (grid_1 == 3)) & (grid_2 == 5)))
        construction_px = int(np.sum((grid_2 == 4) & (grid_1 != 4)))
        regrowth_px = int(np.sum((grid_1 == 5) & ((grid_2 == 2) | (grid_2 == 3))))
        water_gain_px = int(np.sum((grid_1 != 1) & (grid_2 == 1)))
        water_loss_px = int(np.sum((grid_1 == 1) & (grid_2 != 1)))
        ground_activity_px = int(np.sum((grid_1 == grid_2) & ((np.abs(d_ndbi) >= 0.04) | (np.abs(d_ndvi) >= 0.08))))

        changed_px = clearing_px + construction_px + regrowth_px + water_gain_px + water_loss_px + ground_activity_px
        stable_px = max(0, total_px - changed_px)

        # 5. Patch-level change type decision based on Dominant Spectral Shift Vector Analysis (DSSVA)
        mean_d_ndvi = float(np.mean(d_ndvi))
        mean_d_ndwi = float(np.mean(d_ndwi))
        mean_d_ndbi = float(np.mean(d_ndbi))

        primary_type, trans_label, _ = cls.classify_by_dominant_delta(
            delta_ndvi=mean_d_ndvi,
            delta_ndwi=mean_d_ndwi,
            delta_ndbi=mean_d_ndbi,
            dom_t1_name=dom_t1_name,
            dom_t2_name=dom_t2_name,
        )

        # Align dom_t2_name with the dominant physical transition
        if primary_type == "New Construction":
            dom_t2_name = "Built-Up"
        elif primary_type in ("Vegetation Increase", "Vegetation Regrowth"):
            dom_t2_name = "Vegetation"
        elif primary_type in ("Vegetation Decrease", "Land Clearance"):
            dom_t2_name = "Bare Land" if dom_t1_name == "Vegetation" else dom_t2_name
        elif primary_type == "Water Body Expansion":
            dom_t2_name = "Water"
        elif primary_type == "Water Body Recession":
            dom_t2_name = "Bare Land" if dom_t1_name == "Water" else dom_t2_name

        return {
            "dominant_t1_class": dom_t1_name,
            "dominant_t2_class": dom_t2_name,
            "predicted_type": primary_type,
            "transition_label": trans_label,
            "indices_t1": {
                "ndvi": round(float(np.mean(ndvi_1)), 4),
                "ndwi": round(float(np.mean(ndwi_1)), 4),
                "ndbi": round(float(np.mean(ndbi_1)), 4),
            },
            "indices_t2": {
                "ndvi": round(float(np.mean(ndvi_2)), 4),
                "ndwi": round(float(np.mean(ndwi_2)), 4),
                "ndbi": round(float(np.mean(ndbi_2)), 4),
            },
            "deltas": {
                "delta_ndvi": round(float(np.mean(d_ndvi)), 4),
                "delta_ndwi": round(float(np.mean(d_ndwi)), 4),
                "delta_ndbi": round(float(np.mean(d_ndbi)), 4),
            },
            "pixel_stats": {
                "total_pixels": total_px,
                "clearing_px": clearing_px,
                "construction_px": construction_px,
                "regrowth_px": regrowth_px,
                "water_gain_px": water_gain_px,
                "water_loss_px": water_loss_px,
                "ground_activity_px": ground_activity_px,
                "stable_px": stable_px,
                "changed_pixels": changed_px,
                "changed_pct": round(float(changed_px / total_px * 100), 1),
                "t1_composition": {
                    CLASS_MAP[i]: round(float(counts_1[i] / total_px * 100), 1)
                    for i in range(1, 6)
                },
                "t2_composition": {
                    CLASS_MAP[i]: round(float(counts_2[i] / total_px * 100), 1)
                    for i in range(1, 6)
                }
            }
        }


    @staticmethod
    def classify_land_cover(
        ndvi: float,
        ndwi: float,
        ndbi: float,
        bsi: Optional[float] = None
    ) -> Tuple[int, str]:
        """
        Maps (NDVI, NDWI, NDBI, BSI) to standardized Sentinel-2 physical classes:
        1. Water: NDWI >= 0.20 and NDVI < 0.20 (or NDWI > 0.10 and NDWI > NDVI)
        2. Dense Vegetation: NDVI >= 0.40 and NDWI < 0.20
        3. Sparse Vegetation: 0.20 <= NDVI < 0.40 and NDWI < 0.20
        4. Built-Up Area: NDVI < 0.25 and (NDBI >= 0.10 or (bsi is not None and NDBI >= bsi))
        5. Bare Soil: NDVI < 0.25 and (bsi is not None and bsi > NDBI or NDBI < 0.10)
        """
        # 1. Water Body
        if (ndwi >= 0.20 and ndvi < 0.20) or (ndwi > 0.10 and ndwi > ndvi):
            return 1, "Water Body"

        # 2. Dense Vegetation
        if ndvi >= 0.40 and ndwi < 0.20:
            return 2, "Dense Vegetation"

        # 3. Sparse Vegetation
        if 0.20 <= ndvi < 0.40 and ndwi < 0.20:
            return 3, "Sparse Vegetation"

        # 4. Built-Up vs Bare Soil
        if ndvi < 0.25 and ndwi < 0.20:
            if bsi is not None:
                if ndbi >= 0.10 or ndbi >= bsi:
                    return 4, "Built-Up Area"
                else:
                    return 5, "Bare Soil"
            else:
                if ndbi >= 0.10:
                    return 4, "Built-Up Area"
                else:
                    return 5, "Bare Soil"

        # Boundary / residual conditions
        if ndbi >= 0.08:
            return 4, "Built-Up Area"
        elif ndvi >= 0.18:
            return 3, "Sparse Vegetation"
        elif bsi is not None and bsi > 0.0:
            return 5, "Bare Soil"

        return 6, "Mixed Land Cover"

    @classmethod
    def classify_by_dominant_delta(
        cls,
        delta_ndvi: float,
        delta_ndwi: float,
        delta_ndbi: float,
        dom_t1_name: str = "Bare Land",
        dom_t2_name: str = "Bare Land",
        min_threshold: float = 0.02
    ) -> Tuple[str, str, str]:
        """
        Dominant Spectral Shift Vector Analysis (DSSVA).
        Evaluates physical delta magnitudes across NDVI (vegetation canopy),
        NDWI (hydrological/moisture), and NDBI (built-up impervious surface)
        to identify the dominant driver of ground change.

        Returns: (predicted_type, transition_label, interpretation)
        """
        m_ndvi = abs(delta_ndvi)
        m_ndwi = abs(delta_ndwi)
        m_ndbi = abs(delta_ndbi)

        max_delta = max(m_ndvi, m_ndwi, m_ndbi)

        if max_delta < min_threshold:
            pred_type = "Stable Terrain"
            trans_label = f"Stable {dom_t1_name}"
            interp = f"Stable terrain with minimal spectral shift ({dom_t1_name})"
            return pred_type, trans_label, interp

        # Case 1: Hydrological / Moisture dominance (NDWI)
        if m_ndwi >= m_ndvi and m_ndwi >= m_ndbi:
            if delta_ndwi > 0:
                pred_type = "Water Body Expansion"
                trans_label = f"{dom_t1_name} → Water Body" if dom_t1_name not in ("Water", "Water Body") else "Water Body Expansion"
                interp = f"Water / moisture expansion observed (ΔNDWI +{delta_ndwi:.3f})"
            else:
                pred_type = "Water Body Recession"
                trans_label = f"Water Body → {dom_t2_name}" if dom_t1_name in ("Water", "Water Body") else f"{dom_t1_name} Moisture Loss"
                interp = f"Water loss / moisture recession observed (ΔNDWI {delta_ndwi:.3f})"

        # Case 2: Built-Up / Structural dominance (NDBI)
        elif m_ndbi >= m_ndvi and m_ndbi >= m_ndwi:
            if delta_ndbi > 0:
                pred_type = "New Construction"
                trans_label = f"{dom_t1_name} → Built-Up"
                interp = f"New structural construction / built-up expansion (ΔNDBI +{delta_ndbi:.3f})"
            else:
                pred_type = "Built-Up Reduction"
                trans_label = f"Built-Up → {dom_t2_name}" if dom_t1_name in ("Built-Up", "Built-Up Area") else f"Structural Demolition ({dom_t1_name})"
                interp = f"Built-up reduction / demolition observed (ΔNDBI {delta_ndbi:.3f})"

        # Case 3: Vegetation Canopy dominance (NDVI)
        else:
            if delta_ndvi > 0:
                pred_type = "Vegetation Increase"
                if dom_t1_name in ("Vegetation", "Dense Vegetation", "Dense Forest", "Farmland / Crops", "Farmland"):
                    trans_label = "Vegetation Canopy Growth & Greening"
                else:
                    trans_label = f"{dom_t1_name} → Vegetation"
                interp = f"Vegetation increase / greening observed (ΔNDVI +{delta_ndvi:.3f})"
            else:
                # If NDBI is also distinctly positive (e.g. >= 0.05), it's land clearance directly for construction
                if delta_ndbi >= 0.05:
                    pred_type = "New Construction"
                    trans_label = f"{dom_t1_name} → Built-Up"
                    interp = f"Vegetation clearance for construction (ΔNDBI +{delta_ndbi:.3f}, ΔNDVI {delta_ndvi:.3f})"
                else:
                    pred_type = "Vegetation Decrease"
                    if dom_t2_name in ("Vegetation", "Dense Vegetation", "Dense Forest", "Farmland / Crops"):
                        trans_label = "Vegetation Canopy Disturbance & Thinning"
                    else:
                        trans_label = f"Vegetation → {dom_t2_name}" if dom_t1_name in ("Vegetation", "Dense Vegetation", "Dense Forest", "Farmland / Crops") else f"Canopy Clearance ({dom_t1_name})"
                    interp = f"Vegetation decrease / canopy clearance observed (ΔNDVI {delta_ndvi:.3f})"

        return pred_type, trans_label, interp

    @classmethod
    def get_transition_summary(
        cls,
        class_t1_name: str,
        class_t2_name: str,
        delta_ndbi: float = 0.0,
        delta_ndvi: float = 0.0,
        delta_ndwi: float = 0.0,
    ) -> Tuple[str, str]:
        """
        Determines clean, professional transition type and label using Dominant Spectral Shift Vector Analysis.
        """
        pred_type, trans_label, _ = cls.classify_by_dominant_delta(
            delta_ndvi=delta_ndvi,
            delta_ndwi=delta_ndwi,
            delta_ndbi=delta_ndbi,
            dom_t1_name=class_t1_name,
            dom_t2_name=class_t2_name,
        )
        return pred_type, trans_label

    @classmethod
    def evaluate_candidates(
        cls,
        candidates: List[Dict[str, Any]],
        mb_before: np.ndarray,
        mb_after: np.ndarray,
        band_descriptions: Optional[List[str]] = None,
        band_descriptions_b: Optional[List[str]] = None,
        band_descriptions_a: Optional[List[str]] = None,
        stability_threshold: float = 0.002
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Evaluates each candidate patch:
        1. Computes T1 & T2 spectral indices (NDVI, NDWI, NDBI, BSI).
        2. Drops candidates ONLY if all three (|ΔNDVI|, |ΔNDWI|, |ΔNDBI|) < stability_threshold (0.002).
        3. Classifies T1 and T2 states and assigns clean transition labels.
        """
        confirmed = []
        dropped_stable = []

        desc_b = band_descriptions_b or (band_descriptions if (band_descriptions and len(band_descriptions) == mb_before.shape[0]) else None)
        desc_a = band_descriptions_a or (band_descriptions if (band_descriptions and len(band_descriptions) == mb_after.shape[0]) else None)

        for cand in candidates:
            x0, y0, x1, y1 = cand["bbox"]
            crop_b = mb_before[:, y0:y1, x0:x1]
            crop_a = mb_after[:, y0:y1, x0:x1]

            ind_b = cls.compute_patch_indices(crop_b, desc_b)
            ind_a = cls.compute_patch_indices(crop_a, desc_a)

            d_ndvi = round(ind_a["ndvi"] - ind_b["ndvi"], 4)
            d_ndwi = round(ind_a["ndwi"] - ind_b["ndwi"], 4)
            d_ndbi = round(ind_a["ndbi"] - ind_b["ndbi"], 4)
            d_bsi = round(ind_a["bsi"] - ind_b["bsi"], 4)

            # Ultra-strict stability check: only drop if virtually identical (< 0.002)
            if (
                abs(d_ndvi) < stability_threshold
                and abs(d_ndwi) < stability_threshold
                and abs(d_ndbi) < stability_threshold
            ):
                cand_copy = dict(cand)
                cand_copy["is_candidate"] = False
                cand_copy["drop_reason"] = "spectral_invariance_rejected"
                cand_copy["indices_t1"] = ind_b
                cand_copy["indices_t2"] = ind_a
                cand_copy["deltas"] = {
                    "delta_ndvi": d_ndvi,
                    "delta_ndwi": d_ndwi,
                    "delta_ndbi": d_ndbi,
                    "delta_bsi": d_bsi,
                }
                dropped_stable.append(cand_copy)
                continue

            c1_id, c1_name = cls.classify_land_cover(
                ind_b["ndvi"], ind_b["ndwi"], ind_b["ndbi"], ind_b.get("bsi")
            )
            c2_id, c2_name = cls.classify_land_cover(
                ind_a["ndvi"], ind_a["ndwi"], ind_a["ndbi"], ind_a.get("bsi")
            )

            trans_type, trans_label = cls.get_transition_summary(
                c1_name, c2_name, delta_ndbi=d_ndbi, delta_ndvi=d_ndvi, delta_ndwi=d_ndwi
            )

            cand["indices_t1"] = ind_b
            cand["indices_t2"] = ind_a
            cand["deltas"] = {
                "delta_ndvi": d_ndvi,
                "delta_ndwi": d_ndwi,
                "delta_ndbi": d_ndbi,
                "delta_bsi": d_bsi,
            }
            cand["class_t1"] = {"id": c1_id, "name": c1_name}
            cand["class_t2"] = {"id": c2_id, "name": c2_name}
            cand["transition_type"] = trans_type
            cand["transition_label"] = trans_label

            confirmed.append(cand)

        return confirmed, dropped_stable

    @classmethod
    def aggregate_cluster_consensus(
        cls,
        confirmed_candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Groups confirmed candidate patches by cluster_id, computes majority vote
        for T1 class and T2 class, and assigns clean unified transition identity.
        """
        cluster_groups: Dict[int, List[Dict[str, Any]]] = {}

        for cand in confirmed_candidates:
            cid = cand.get("cluster_id")
            if cid is not None:
                cluster_groups.setdefault(cid, []).append(cand)

        cluster_results = {}
        for cid, patches in cluster_groups.items():
            t1_classes = [p["class_t1"]["name"] for p in patches if "class_t1" in p]
            t2_classes = [p["class_t2"]["name"] for p in patches if "class_t2" in p]

            majority_t1 = Counter(t1_classes).most_common(1)[0][0] if t1_classes else "Unknown"
            majority_t2 = Counter(t2_classes).most_common(1)[0][0] if t2_classes else "Unknown"

            avg_d_ndvi = round(float(np.mean([p["deltas"]["delta_ndvi"] for p in patches])), 4)
            avg_d_ndwi = round(float(np.mean([p["deltas"]["delta_ndwi"] for p in patches])), 4)
            avg_d_ndbi = round(float(np.mean([p["deltas"]["delta_ndbi"] for p in patches])), 4)
            avg_dist = round(float(np.mean([p["distance"] for p in patches])), 4)

            # Area calculation: 64x64 pixels at 10m Ground Sample Distance (GSD) = 409,600 m2 per patch
            area_m2 = len(patches) * 64 * 64 * 100

            trans_type, trans_label, interpretation = cls.classify_by_dominant_delta(
                delta_ndvi=avg_d_ndvi,
                delta_ndwi=avg_d_ndwi,
                delta_ndbi=avg_d_ndbi,
                dom_t1_name=majority_t1,
                dom_t2_name=majority_t2,
            )

            cluster_results[str(cid)] = {
                "cluster_id": cid,
                "num_patches": len(patches),
                "area_m2": area_m2,
                "majority_class_t1": majority_t1,
                "majority_class_t2": majority_t2,
                "dominant_t1_class": majority_t1,
                "dominant_t2_class": majority_t2,
                "transition_type": trans_type,
                "transition_label": trans_label,
                "cluster_transition": f"{majority_t1} → {majority_t2}",
                "interpretation": interpretation,
                "mean_delta_ndvi": avg_d_ndvi,
                "mean_delta_ndwi": avg_d_ndwi,
                "mean_delta_ndbi": avg_d_ndbi,
                "cluster_averages": {
                    "mean_distance": avg_dist,
                    "avg_delta_ndvi": avg_d_ndvi,
                    "avg_delta_ndwi": avg_d_ndwi,
                    "avg_delta_ndbi": avg_d_ndbi,
                },
                "patches": [
                    {
                        "patch_id": p["patch_id"],
                        "row": p["row"],
                        "col": p["col"],
                        "class_t1": p.get("class_t1", {}).get("name"),
                        "class_t2": p.get("class_t2", {}).get("name"),
                        "transition": p.get("transition_label"),
                        "ndvi_t1": p.get("indices_t1", {}).get("ndvi"),
                        "ndvi_t2": p.get("indices_t2", {}).get("ndvi"),
                        "ndbi_t1": p.get("indices_t1", {}).get("ndbi"),
                        "ndbi_t2": p.get("indices_t2", {}).get("ndbi"),
                    }
                    for p in patches
                ]
            }

        return cluster_results
