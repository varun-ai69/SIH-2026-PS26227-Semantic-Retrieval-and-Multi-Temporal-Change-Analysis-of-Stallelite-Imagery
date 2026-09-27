import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.api.routers.coverage import router as coverage_router
from backend.api.routers.ingest import router as ingest_router
from backend.api.routers.archive import router as archive_router
from backend.api.routers.search import router as search_router
from backend.api.routers.change import router as change_router
from backend.api.routers.discovery import router as discovery_router
from backend.api.routers.chat import router as chat_router
from backend.api.routers.review import router as review_router

app = FastAPI(
    title="Satellite Imagery Semantic Retrieval & Change Detection API",
    description="Backend service for semantic retrieval and multi-temporal change analysis (PS SIH-26227 for Indian Army DGIS).",
    version="1.0.0"
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(coverage_router)
app.include_router(ingest_router)
app.include_router(archive_router)
app.include_router(search_router)
app.include_router(change_router)
app.include_router(discovery_router)
app.include_router(chat_router)
app.include_router(review_router)

# Mount Data & Static directories if they exist
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
data_dir = REPO_ROOT / "data"
frontend_dir = REPO_ROOT / "frontend"

data_dir.mkdir(parents=True, exist_ok=True)
frontend_dir.mkdir(parents=True, exist_ok=True)

app.mount("/data", StaticFiles(directory=str(data_dir)), name="data")
app.mount("/frontend", StaticFiles(directory=str(frontend_dir)), name="frontend")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/")
def read_root():
    index_file = frontend_dir / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "status": "online",
        "service": "Satellite Imagery Semantic Retrieval & Change Detection API",
        "docs": "/docs"
    }


@app.get("/retrieval")
@app.get("/retrieval.html")
def read_retrieval_page():
    retrieval_file = frontend_dir / "retrieval.html"
    if retrieval_file.exists():
        return FileResponse(str(retrieval_file))
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/change")
@app.get("/change.html")
def read_change_page():
    change_file = frontend_dir / "change.html"
    if change_file.exists():
        return FileResponse(str(change_file))
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/clustering")
@app.get("/clustering.html")
def read_clustering_page():
    clustering_file = frontend_dir / "clustering.html"
    if clustering_file.exists():
        return FileResponse(str(clustering_file))
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/review")
@app.get("/review.html")
def read_review_page():
    review_file = frontend_dir / "review.html"
    if review_file.exists():
        return FileResponse(str(review_file))
    return FileResponse(str(frontend_dir / "index.html"))


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


# ============================================================
# SYSTEM MODE & AIR-GAPPED OFFLINE MANAGEMENT
# ============================================================
from pydantic import BaseModel, Field
from backend.services.system_mode import (
    is_offline_mode,
    set_offline_mode,
    get_system_mode_status
)

class SystemModeUpdateRequest(BaseModel):
    offline_mode: bool = Field(..., description="True for Air-Gapped Offline Mode, False for Online Ingestion Mode")

@app.get("/api/v1/system/mode", tags=["System"])
def get_system_mode():
    """Retrieve current operational network mode (Online vs Air-Gapped Offline)."""
    return get_system_mode_status()

@app.post("/api/v1/system/mode", tags=["System"])
def update_system_mode(req: SystemModeUpdateRequest):
    """Toggle between Online Ingestion Mode and strict Air-Gapped Offline Mode at runtime."""
    set_offline_mode(req.offline_mode)
    return get_system_mode_status()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("backend.api.main:app", host="0.0.0.0", port=port, reload=True)
