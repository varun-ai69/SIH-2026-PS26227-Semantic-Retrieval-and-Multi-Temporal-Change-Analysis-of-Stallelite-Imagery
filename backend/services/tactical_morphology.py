"""
backend/services/tactical_morphology.py
======================================
Tactical Spatial Morphology & Multi-Temporal Narrative Synthesis
================================================================
Aligns with NetSight military-grade spatial dynamics:
1. Classifies localized cluster transitions into directional morphology:
   - APPEARANCE: Novel target feature or structural foundation emergence where no prior footprint existed.
   - DISAPPEARANCE: Complete elimination or demolition of an existing feature.
   - EXPANSION: Outward growth or perimeter enlargement of an existing cluster (Area growth >= 15%).
   - CONTRACTION: Perimeter shrinkage or receding boundary (Area decrease >= 15%).
2. Generates an executive Whole-Window Intelligence Summary synthesizing the complete
   multi-year timeline trajectory.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger("tactical_morphology")

TACTICAL_DYNAMICS_META = {
    "APPEARANCE": {
        "label": "APPEARANCE",
        "color": "#ef4444",
        "bg": "rgba(239, 68, 68, 0.15)",
        "border": "#ef4444",
        "icon": "plus-circle",
        "desc": "Emergence of new structural, transport, or physical asset"
    },
    "EXPANSION": {
        "label": "EXPANSION",
        "color": "#f59e0b",
        "bg": "rgba(245, 158, 11, 0.15)",
        "border": "#f59e0b",
        "icon": "maximize-2",
        "desc": "Perimeter enlargement or outward footprint growth"
    },
    "CONTRACTION": {
        "label": "CONTRACTION",
        "color": "#3b82f6",
        "bg": "rgba(59, 130, 246, 0.15)",
        "border": "#3b82f6",
        "icon": "minimize-2",
        "desc": "Perimeter shrinkage or receding physical boundary"
    },
    "DISAPPEARANCE": {
        "label": "DISAPPEARANCE",
        "color": "#94a3b8",
        "bg": "rgba(148, 163, 184, 0.15)",
        "border": "#94a3b8",
        "icon": "minus-circle",
        "desc": "Structural demolition, vegetation clearance, or asset removal"
    }
}


def classify_tactical_dynamic(
    cluster: Dict[str, Any],
    prior_cluster: Optional[Dict[str, Any]] = None,
    step_index: int = 1
) -> Dict[str, Any]:
    """
    Classifies a confirmed change cluster into one of four tactical dynamics:
    APPEARANCE, EXPANSION, CONTRACTION, DISAPPEARANCE.
    
    Args:
        cluster: Current change cluster metadata (bounding box, area, deltas, transition_type).
        prior_cluster: Spatially corresponding cluster from the previous temporal step (if available).
        step_index: 1 for initial change onset; > 1 for subsequent evolution steps.
    """
    deltas = cluster.get("deltas") or cluster.get("cluster_averages") or {}
    d_ndbi = float(deltas.get("delta_ndbi") if deltas.get("delta_ndbi") is not None else (
        deltas.get("avg_delta_ndbi") if deltas.get("avg_delta_ndbi") is not None else (
            cluster.get("mean_delta_ndbi") or 0.0
        )
    ))
    d_ndvi = float(deltas.get("delta_ndvi") if deltas.get("delta_ndvi") is not None else (
        deltas.get("avg_delta_ndvi") if deltas.get("avg_delta_ndvi") is not None else (
            cluster.get("mean_delta_ndvi") or 0.0
        )
    ))
    d_ndwi = float(deltas.get("delta_ndwi") if deltas.get("delta_ndwi") is not None else (
        deltas.get("avg_delta_ndwi") if deltas.get("avg_delta_ndwi") is not None else (
            cluster.get("mean_delta_ndwi") or 0.0
        )
    ))
    
    indices_t1 = cluster.get("indices_t1") or cluster.get("t1_indices") or {}
    t1_ndbi = float(indices_t1.get("ndbi") or cluster.get("mean_t1_ndbi") or 0.0)
    t1_ndvi = float(indices_t1.get("ndvi") or cluster.get("mean_t1_ndvi") or 0.0)
    
    current_area = float(cluster.get("area_sq_m") or cluster.get("area_m2") or cluster.get("num_patches", 1) * 4096.0)
    prior_area = float(prior_cluster.get("area_sq_m") or prior_cluster.get("area_m2") or prior_cluster.get("num_patches", 1) * 4096.0) if prior_cluster else None

    dynamic = "APPEARANCE"
    confidence = 0.88
    rationale = ""

    # Rule 1: Multi-Step Spatial Growth / Shrinkage Tracking (when prior step cluster exists)
    if prior_area is not None and prior_area > 0:
        area_ratio = current_area / prior_area
        if area_ratio >= 1.15:
            dynamic = "EXPANSION"
            confidence = min(0.98, 0.85 + (area_ratio - 1.0) * 0.1)
            growth_pct = (area_ratio - 1.0) * 100.0
            rationale = f"Cluster footprint expanded by +{growth_pct:.1f}% ({current_area:.0f} m² vs {prior_area:.0f} m² in prior epoch)."
        elif area_ratio <= 0.85:
            dynamic = "CONTRACTION"
            confidence = min(0.98, 0.85 + (1.0 - area_ratio) * 0.1)
            shrink_pct = (1.0 - area_ratio) * 100.0
            rationale = f"Cluster footprint contracted by -{shrink_pct:.1f}% ({current_area:.0f} m² vs {prior_area:.0f} m² in prior epoch)."
        else:
            if d_ndbi > 0:
                dynamic = "EXPANSION"
                rationale = f"Consolidated footprint area ({current_area:.0f} m²) with structural densification (ΔNDBI +{d_ndbi:.3f})."
            else:
                dynamic = "CONTRACTION"
                rationale = f"Footprint stabilized with receding physical intensity (ΔNDBI {d_ndbi:.3f})."

    # Rule 2: Single-Step / Onset Step Classification
    else:
        # Check Disappearance (structural demolition or water loss)
        if d_ndbi <= -0.04 and t1_ndbi >= 0.02:
            dynamic = "DISAPPEARANCE"
            confidence = 0.92
            rationale = f"Structural asset present in T1 (NDBI {t1_ndbi:.3f}) was demolished or removed in T2 (ΔNDBI {d_ndbi:.3f})."
        elif d_ndwi <= -0.08:
            dynamic = "CONTRACTION" if step_index > 1 else "DISAPPEARANCE"
            confidence = 0.90
            rationale = f"Surface water or moisture reservoir receded significantly (ΔNDWI {d_ndwi:.3f})."
        elif d_ndvi <= -0.12 and d_ndbi < 0.02:
            dynamic = "CONTRACTION" if step_index > 1 else "DISAPPEARANCE"
            confidence = 0.89
            rationale = f"Vegetation canopy clear-cut and biomass loss observed (ΔNDVI {d_ndvi:.3f})."
        elif d_ndbi >= 0.035 or (t1_ndbi < 0.0 and d_ndbi > 0.02):
            dynamic = "APPEARANCE"
            confidence = 0.95
            rationale = f"New structural built-up footprint emerged on previously unpaved terrain (ΔNDBI +{d_ndbi:.3f})."
        elif d_ndvi >= 0.08:
            dynamic = "APPEARANCE"
            confidence = 0.91
            rationale = f"Active vegetation canopy densification and crop/foliage biomass greening (ΔNDVI +{d_ndvi:.3f})."
        elif d_ndwi >= 0.08:
            dynamic = "APPEARANCE"
            confidence = 0.88
            rationale = f"Surface moisture accumulation and water feature emergence detected (ΔNDWI +{d_ndwi:.3f})."
        else:
            dynamic = "APPEARANCE"
            confidence = 0.86
            rationale = f"Distinct physical land-cover disturbance detected with confirmed spectral differentiation."

    meta = TACTICAL_DYNAMICS_META.get(dynamic, TACTICAL_DYNAMICS_META["APPEARANCE"])

    return {
        "dynamic": dynamic,
        "label": meta["label"],
        "color": meta["color"],
        "bg": meta["bg"],
        "border": meta["border"],
        "confidence": round(confidence, 2),
        "rationale": rationale
    }


def synthesize_whole_window_narrative(
    site_key: str,
    total_clean_epochs: int,
    dates: List[str],
    steps: List[Dict[str, Any]],
    spectral_deltas_global: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Synthesizes a cohesive, military-grade macro narrative describing the
    multi-year physical evolution of the site footprint across the entire time window.
    """
    if not dates:
        return {
            "headline": "Insufficient temporal observations for narrative synthesis.",
            "full_narrative": "No valid observations recorded."
        }

    start_date = dates[0]
    end_date = dates[-1]
    start_year = start_date[:4]
    end_year = end_date[:4]

    total_steps = len(steps)
    if total_steps == 0:
        return {
            "headline": f"Stable Terrain Footprint ({start_year} – {end_year})",
            "full_narrative": (
                f"Continuous satellite monitoring across {total_clean_epochs} clean observations from "
                f"{start_date} to {end_date} confirms ground stability with zero persistent physical anomalies."
            ),
            "status": "STABLE"
        }

    # Gather dynamics and transitions across all steps
    dynamics_tally = {}
    transitions_list = []
    total_impact_area_sq_m = 0.0

    for step in steps:
        s_date_b = step.get("date_before", "")
        s_date_a = step.get("date_after", "")
        cands = step.get("candidates") or []
        clusters = step.get("clusters") or []
        
        for c in clusters:
            dyn = c.get("tactical_dynamic", {}).get("dynamic", "APPEARANCE")
            dynamics_tally[dyn] = dynamics_tally.get(dyn, 0) + 1
            area_m2 = float(c.get("area_sq_m") or c.get("area_m2") or 0.0)
            total_impact_area_sq_m += area_m2
            
            ttype = c.get("transition_type") or "Ground Transformation"
            transitions_list.append(f"{dyn.title()} of {ttype} between {s_date_b} and {s_date_a}")

    total_impact_ha = total_impact_area_sq_m / 10000.0

    # Build clear chronological story
    step_descriptions = []
    for idx, step in enumerate(steps):
        s_num = idx + 1
        d_b = step.get("date_before", "")
        d_a = step.get("date_after", "")
        role = step.get("role", "Evolution")
        clusters = step.get("clusters") or []
        
        if not clusters:
            step_descriptions.append(
                f"Step {s_num} ({d_b} → {d_a}): Terrain stabilized with minor background fluctuations."
            )
            continue
            
        prim_cluster = clusters[0]
        dyn = prim_cluster.get("tactical_dynamic", {}).get("dynamic", "APPEARANCE")
        ttype = prim_cluster.get("transition_type", "Transformation")
        deltas = prim_cluster.get("deltas", {})
        d_ndbi = deltas.get("delta_ndbi", 0.0)
        d_ndvi = deltas.get("delta_ndvi", 0.0)

        step_descriptions.append(
            f"Step {s_num} [{role}] ({d_b} → {d_a}): {dyn} of {ttype} "
            f"spanning {len(clusters)} spatial cluster(s) (ΔNDBI: {d_ndbi:+.3f}, ΔNDVI: {d_ndvi:+.3f})."
        )

    # Determine predominant status
    if "APPEARANCE" in dynamics_tally and "EXPANSION" in dynamics_tally:
        headline = f"Progressive Structural Development ({start_year} – {end_year})"
        overall_status = "ACTIVE_DEVELOPMENT"
    elif "APPEARANCE" in dynamics_tally:
        headline = f"New Physical Footprint Emergence ({start_year} – {end_year})"
        overall_status = "NEW_EMERGENCE"
    elif "EXPANSION" in dynamics_tally:
        headline = f"Active Perimeter Expansion ({start_year} – {end_year})"
        overall_status = "EXPANDING"
    elif "DISAPPEARANCE" in dynamics_tally:
        headline = f"Physical Demolition & Ground Recession ({start_year} – {end_year})"
        overall_status = "RECESSION"
    else:
        headline = f"Multi-Temporal Terrain Transformation ({start_year} – {end_year})"
        overall_status = "TRANSFORMED"

    narrative_body = (
        f"Multi-temporal analysis of footprint {site_key} across {total_clean_epochs} validated observations "
        f"({start_date} to {end_date}) identified {len(steps)} sequential transition interval(s) with an "
        f"aggregate ground impact of {total_impact_ha:.2f} hectares ({total_impact_area_sq_m:,.0f} m²). "
        + " ".join(step_descriptions)
    )

    return {
        "headline": headline,
        "status": overall_status,
        "start_date": start_date,
        "end_date": end_date,
        "total_epochs": total_clean_epochs,
        "total_steps": total_steps,
        "total_impact_hectares": round(total_impact_ha, 2),
        "total_impact_sq_m": round(total_impact_area_sq_m, 0),
        "dynamics_breakdown": dynamics_tally,
        "full_narrative": narrative_body
    }
