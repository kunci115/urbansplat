"""Central configuration. All values overridable via environment."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="URBANSPLAT_", env_file=".env", extra="ignore")

    # --- Core ---
    # When true, GPU/external tools (COLMAP, gsplat) are skipped and stubbed.
    # Lets the whole pipeline run end-to-end on a CPU-only / no-COLMAP box.
    dry_run: bool = True
    work_dir: str = "/tmp/urbansplat"

    # --- Database ---
    database_url: str = "postgresql+psycopg://urbansplat:urbansplat@postgres:5432/urbansplat"

    # --- Redis / Celery ---
    redis_url: str = "redis://redis:6379/0"

    # --- Object storage (MinIO / S3) ---
    s3_endpoint: str = "minio:9000"
    s3_access_key: str = "urbansplat"
    s3_secret_key: str = "urbansplat"
    s3_secure: bool = False
    s3_bucket: str = "urbansplat"

    # --- Pipeline params ---
    frame_sample_fps: float = 2.0          # frames extracted per second of video
    blur_threshold: float = 60.0           # Laplacian variance; below = drop frame
    train_iterations: int = 30000
    cubemap_face_size: int = 1024          # px per cubemap face for panorama_sfm

    @property
    def celery_broker(self) -> str:
        return self.redis_url

    @property
    def celery_backend(self) -> str:
        return self.redis_url


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
