"""Stage 1 — frame extraction.

Sample frames from the 360 video at a target rate, then drop blurry frames using
the variance-of-Laplacian sharpness metric. Sharp, well-spaced frames are what SfM
needs; motion-blurred frames poison pose estimation downstream.
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


def extract_frames(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        # Produce a handful of stub frames so downstream stages have inputs.
        for i in range(8):
            (ctx.frames_dir / f"frame_{i:04d}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        ctx.metrics["frames_kept"] = 8
        ctx.metrics["frames_dropped"] = 0
        log.append("[dry-run] wrote 8 stub frames")
        return

    # 1. Sample frames with ffmpeg at the configured fps.
    pattern = str(ctx.frames_dir / "frame_%05d.jpg")
    run_command(
        [
            "ffmpeg", "-hide_banner", "-i", str(ctx.source_video),
            "-vf", f"fps={settings.frame_sample_fps}",
            "-q:v", "2", pattern,
        ],
        log,
    )

    # 2. Blur filter — drop frames below the sharpness threshold (skipped if no opencv).
    kept, dropped = 0, 0
    for frame in sorted(ctx.frames_dir.glob("frame_*.jpg")):
        sharpness = _measure_blur(frame)
        if sharpness is not None and sharpness < settings.blur_threshold:
            frame.unlink()
            dropped += 1
        else:
            kept += 1

    ctx.metrics["frames_kept"] = kept
    ctx.metrics["frames_dropped"] = dropped
    log.append(f"kept {kept} frames, dropped {dropped} blurry")

    if kept < 8:
        raise StageError(
            f"only {kept} sharp frames extracted — input too short/blurry for reconstruction"
        )
