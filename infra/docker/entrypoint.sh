#!/bin/bash
set -e

echo "=========================================================="
echo "      Starting CANOPUS Defense & EO Intelligence Stack    "
echo "=========================================================="

echo "Waiting for PostgreSQL at ${POSTGRES_HOST:-postgres}:${POSTGRES_PORT:-5432}..."
while ! python -c "import psycopg2, os; psycopg2.connect(host=os.getenv('POSTGRES_HOST', 'postgres'), port=int(os.getenv('POSTGRES_PORT', '5432')), dbname=os.getenv('POSTGRES_DB', 'eo_archive'), user=os.getenv('POSTGRES_USER', 'eo_admin'), password=os.getenv('POSTGRES_PASSWORD', 'eo_password'))" 2>/dev/null; do
  sleep 1
done
echo " -> PostgreSQL is ready!"

echo "Waiting for Qdrant at ${QDRANT_HOST:-qdrant}:${QDRANT_PORT:-6333}..."
while ! python -c "import os; from qdrant_client import QdrantClient; client = QdrantClient(host=os.getenv('QDRANT_HOST', 'qdrant'), port=int(os.getenv('QDRANT_PORT', '6333')), check_compatibility=False); client.get_collections()" 2>/dev/null; do
  sleep 1
done
echo " -> Qdrant is ready!"

echo "Applying PostgreSQL schema migrations..."
python -c "from backend.ingestion.db_writer import ensure_schema_migrated; ensure_schema_migrated()" || true

echo "Ensuring Qdrant collection 'tile_embeddings' exists..."
python -c "from backend.services.vector_store import get_qdrant_client, ensure_collection_exists; ensure_collection_exists(get_qdrant_client())" || true

echo "=========================================================="
echo "    Launching FastAPI Server & CANOPUS Web UI on Port 8000"
echo "=========================================================="
exec uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
