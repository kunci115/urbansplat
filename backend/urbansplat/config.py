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
    # 360 → perspective reprojection (equirectangular input only).
    views_per_frame: int = 8               # perspective views sampled around each pano
    perspective_yaw_offset: float = 0.0    # yaw of first view (90 = side-on, best parallax)
    perspective_fov: float = 90.0          # FOV (deg) of each reprojected view
    perspective_size: int = 1440           # px (square) per reprojected view (sharper text)

    # Dynamic-object masking (people/vehicles) — removes floaters + cleans SfM.
    masking_enabled: bool = True
    seg_model: str = "/app/models/yolo11m-seg.pt"   # pre-downloaded in the GPU image
    seg_classes: str = "0,1,2,3,5,6,7"     # COCO: person,bicycle,car,motorcycle,bus,train,truck
    mask_dilate_px: int = 12               # grow masks to cover edges/shadows
    seg_conf: float = 0.25                 # detection confidence threshold

    # Video inpainting (ProPainter) — fill masked dynamic regions with temporally
    # consistent background so they don't leave holes/noise in the splat.
    inpaint_enabled: bool = True
    propainter_dir: str = "/opt/ProPainter"
    inpaint_proc_size: int = 720           # ProPainter processing resolution (VRAM cap)
    inpaint_subvideo: int = 20             # frames per ProPainter chunk (lower = less VRAM)

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
