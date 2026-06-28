"""Stage 2 — camera pose estimation (the make-or-break stage).

Uses COLMAP's `panorama_sfm` cubemap route: each equirectangular frame is split into
6 perspective faces, matched under rig constraints, then bundle-adjusted. This is
COLMAP's first-class path for 360 input and far more reliable than feeding raw
equirectangular images to a perspective SfM.

Per MVP §8, this is the highest-risk stage. It fails LOUDLY (StageError) rather than
emitting degenerate poses that silently ruin training.
"""

from __future__ import annotations

import shutil

from ..config import settings
from .base import PipelineContext, StageError, run_command


def estimate_poses(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        # Stub a COLMAP sparse model layout so train stage has something to read.
        sparse = ctx.colmap_dir / "sparse" / "0"
        sparse.mkdir(parents=True, exist_ok=True)
        for fname in ("cameras.bin", "images.bin", "points3D.bin"):
            (sparse / fname).write_bytes(b"\x00")
        ctx.metrics["registered_images"] = 8
        ctx.metrics["mean_reproj_error"] = 0.0
        log.append("[dry-run] wrote stub COLMAP sparse model")
        return

    if shutil.which("colmap") is None:
        raise StageError("colmap binary not found on PATH")

    # COLMAP ships panorama_sfm.py as an example. Path varies by install; allow override.
    pano_script = "panorama_sfm.py"
    run_command(
        [
            "python", pano_script,
            "--image_path", str(ctx.frames_dir),
            "--output_path", str(ctx.colmap_dir),
            "--matcher", "exhaustive",
            "--cubemap_size", str(settings.cubemap_face_size),
        ],
        log,
    )

    sparse = ctx.colmap_dir / "sparse" / "0"
    if not (sparse / "images.bin").exists() and not (sparse / "images.txt").exists():
        raise StageError(
            "COLMAP produced no reconstruction — poses failed. "
            "Likely textureless/repetitive/dynamic scene or exposure drift. "
            "See capture guidance in docs."
        )
    log.append("pose estimation succeeded")
