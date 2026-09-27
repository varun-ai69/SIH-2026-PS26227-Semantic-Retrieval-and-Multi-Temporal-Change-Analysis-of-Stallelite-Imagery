"""
backend/services/system_mode.py
===============================
Sovereign Network & Air-Gapped Mode Management Service.

Provides runtime detection and state toggling between:
  1. ONLINE MODE: Allows live Sentinel-2 STAC and Maxar Wayback external tile fetching.
  2. AIR-GAPPED OFFLINE MODE: Disables all external outbound API/tile requests.
     Core local AI capabilities (RemoteCLIP Semantic Retrieval, HDBSCAN Clustering,
     Siamese Multi-Temporal Change Detection, and Analyst Feedback) remain 100% active.
"""

import os
import threading
import logging
from typing import Dict, Any

logger = logging.getLogger("system_mode")

_LOCK = threading.Lock()
# Initialize from environment variable (default: False / online)
_initial_env_val = os.getenv("OFFLINE_MODE", "false").strip().lower()
_OFFLINE_MODE: bool = _initial_env_val in ("true", "1", "yes", "on")


def is_offline_mode() -> bool:
    """Return True if system is currently locked down in Air-Gapped Offline Mode."""
    with _LOCK:
        return _OFFLINE_MODE


def set_offline_mode(offline: bool) -> bool:
    """
    Update runtime Air-Gapped Offline mode.
    Returns the new state.
    """
    global _OFFLINE_MODE
    with _LOCK:
        _OFFLINE_MODE = bool(offline)
        # Also sync to process environment so subprocesses/workers inherit
        os.environ["OFFLINE_MODE"] = "true" if _OFFLINE_MODE else "false"
        if _OFFLINE_MODE:
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
        else:
            os.environ["HF_HUB_OFFLINE"] = "0"
            os.environ["TRANSFORMERS_OFFLINE"] = "0"
            
        logger.info(f"[SYSTEM_MODE] Network Mode toggled: OFFLINE_MODE={_OFFLINE_MODE}")
        return _OFFLINE_MODE


def get_system_mode_status() -> Dict[str, Any]:
    """Return comprehensive tactical status packet for frontend UI and telemetry."""
    offline = is_offline_mode()
    return {
        "offline_mode": offline,
        "status": "offline" if offline else "online",
        "mode_label": "AIR-GAPPED OFFLINE" if offline else "ONLINE (INGEST READY)",
        "can_ingest_external": not offline,
        "description": (
            "System is operating in Air-Gapped Defense Mode. External satellite tile discovery is disabled. "
            "Local database queries, semantic search, clustering, and change analysis are 100% operational."
            if offline else
            "System is operating in Online Mode. Ready to discover and ingest satellite imagery from Sentinel-2 STAC and Maxar."
        ),
        "modules": {
            "semantic_retrieval": {"status": "operational", "local_only": True},
            "clustering_discovery": {"status": "operational", "local_only": True},
            "change_detection": {"status": "operational", "local_only": True},
            "analyst_review": {"status": "operational", "local_only": True},
            "local_file_ingest": {"status": "operational", "local_only": True},
            "stac_sentinel_ingest": {"status": "paused" if offline else "operational", "local_only": False},
            "maxar_wayback_ingest": {"status": "paused" if offline else "operational", "local_only": False},
            "external_basemaps": {"status": "paused" if offline else "operational", "local_only": False}
        }
    }
