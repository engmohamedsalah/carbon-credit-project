# Backend + ML image for a container host (Railway / Render / Fly).
# Build context = repo root so ml/ and the model weights are included.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    DATABASE_PATH=/data/carbon_credits.db

# libgomp1 is required by torch; the rest are small runtime libs for rasterio wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libexpat1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching.
COPY backend/requirements.txt backend/requirements-ml.txt ./backend/
RUN pip install --no-cache-dir -r backend/requirements.txt -r backend/requirements-ml.txt

# App code + ML package (weights are git-LFS; ensure `git lfs pull` ran before building).
COPY backend/ ./backend/
COPY ml/ ./ml/

# SQLite lives on a mounted persistent volume at /data.
VOLUME ["/data"]
WORKDIR /app/backend
EXPOSE 8000

# Host injects $PORT (Render/Railway); default 8000 locally.
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
