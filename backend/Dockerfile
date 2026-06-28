# urbansplat backend (api + worker share this image).
# NOTE: this image carries the orchestration layer. COLMAP + gsplat (the GPU tools)
# are expected on PATH for real runs — add them via a GPU base image / install step
# and set URBANSPLAT_DRY_RUN=0. In dry-run the image needs no GPU tooling.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# ffmpeg for frame extraction; libgl for opencv.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY urbansplat ./urbansplat

EXPOSE 8000
CMD ["uvicorn", "urbansplat.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
