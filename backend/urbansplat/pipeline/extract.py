"""Stage 1 — frame extraction (+ 360 → perspective conversion).

Sample frames from the video, then:
  * If the footage is equirectangular 360 (≈2:1 aspect), reproject each frame into
    several overlapping perspective ("pinhole") views. COLMAP/SfM expects perspective
    images — feeding raw equirectangular frames in directly fails on the projection's
    distortion. This is the cubemap-style workaround from the research notes.
  * Otherwise keep the frames as-is (normal perspective capture).

Finally drop blurry frames via the variance-of-Laplacian sharpness metric.
"""

from __future__ import annotations

from pathlib import Path

from ..config import settings
from .base import PipelineContext, StageError, run_command


def _measure_blur(image_path: Path) -> float | None:
    """Sharpness via variance-of-Laplacian. Returns None if opencv is unavailable."""
    try:
        import cv2
    except ImportError:
        return None
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _is_equirectangular(image_path: Path) -> bool:
    try:
        import cv2
    except ImportError:
        return False
    img = cv2.imread(str(image_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    return 1.8 <= (w / h) <= 2.2  # equirectangular panoramas are 2:1


def _equirect_to_perspectives(src: Path, dst_dir: Path, stem: str) -> int:
    """Reproject one equirectangular frame into N perspective views around the sphere."""
    import cv2
    import py360convert

    equ = cv2.imread(str(src))
    if equ is None:
        return 0
    size = settings.perspective_size
    fov = settings.perspective_fov
    n = settings.views_per_frame
    written = 0
    for i in range(n):
        yaw = (360.0 / n) * i
        persp = py360convert.e2p(
            equ, fov_deg=(fov, fov), u_deg=yaw, v_deg=0.0, out_hw=(size, size)
        )
        cv2.imwrite(str(dst_dir / f"{stem}_v{i:02d}.jpg"), persp)
        written += 1
    return written


def extract_frames(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        for i in range(8):
            (ctx.frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        ctx.metrics["frames_kept"] = 8
        ctx.metrics["frames_dropped"] = 0
        log.append("[dry-run] wrote 8 stub frames")
        return

    # 1. Sample raw frames from the video at the configured fps.
    raw_dir = ctx.work / "raw_frames"
    raw_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ffmpeg", "-hide_banner", "-i", str(ctx.source_video),
            "-vf", f"fps={settings.frame_sample_fps}", "-q:v", "2",
            str(raw_dir / "raw_%05d.jpg"),
        ],
        log,
    )
    raw = sorted(raw_dir.glob("raw_*.jpg"))
    if not raw:
        raise StageError("no frames extracted from video")

    # 2. Equirectangular? reproject to perspective views; else use frames directly.
    if _is_equirectangular(raw[0]):
        total = 0
        for f in raw:
            total += _equirect_to_perspectives(f, ctx.frames_dir, f.stem)
        ctx.metrics["input_kind"] = "equirectangular_360"
        ctx.metrics["perspective_views"] = total
        log.append(f"360 input: {len(raw)} frames → {total} perspective views "
                   f"({settings.views_per_frame}/frame, {settings.perspective_fov}° FOV)")
    else:
        for f in raw:
            f.replace(ctx.frames_dir / f"{f.stem}.jpg")
        ctx.metrics["input_kind"] = "perspective"
        log.append(f"perspective input: {len(raw)} frames")

    # 3. Blur filter (skipped if opencv unavailable).
    kept, dropped = 0, 0
    for frame in sorted(ctx.frames_dir.glob("*.jpg")):
        sharpness = _measure_blur(frame)
        if sharpness is not None and sharpness < settings.blur_threshold:
            frame.unlink()
            dropped += 1
        else:
            kept += 1

    ctx.metrics["frames_kept"] = kept
    ctx.metrics["frames_dropped"] = dropped
    log.append(f"kept {kept} images, dropped {dropped} blurry")

    if kept < 8:
        raise StageError(
            f"only {kept} sharp images after extraction — input too short/blurry/static"
        )
