"""Stage 2 — camera pose estimation (the make-or-break stage).

Real mode runs nerfstudio's `ns-process-data`, which drives COLMAP and emits a
nerfstudio dataset (transforms.json + undistorted images) the trainer consumes.

Per MVP §8 this is the highest-risk stage. It fails LOUDLY (StageError) rather than
emitting degenerate poses that silently ruin training.
"""

from __future__ import annotations

import json

from ..config import settings
from .base import PipelineContext, StageError, run_command


def estimate_poses(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        ctx.processed_dir.mkdir(parents=True, exist_ok=True)
        (ctx.processed_dir / "transforms.json").write_text(json.dumps({"frames": [{}] * 8}))
        ctx.metrics["registered_images"] = 8
        log.append("[dry-run] wrote stub nerfstudio dataset")
        return

    # ns-process-data: extract features, match, run COLMAP, undistort, write transforms.json.
    run_command(
        [
            "ns-process-data", "images",
            "--data", str(ctx.frames_dir),
            "--output-dir", str(ctx.processed_dir),
            "--matching-method", "exhaustive",
        ],
        log,
    )

    transforms = ctx.processed_dir / "transforms.json"
    if not transforms.exists():
        raise StageError(
            "COLMAP produced no reconstruction — poses failed. "
            "Likely textureless/repetitive/dynamic scene, too little camera motion, "
            "or exposure drift. See capture guidance in docs."
        )
    try:
        registered = len(json.loads(transforms.read_text()).get("frames", []))
    except Exception:
        registered = 0
    ctx.metrics["registered_images"] = registered
    if registered < 8:
        raise StageError(f"only {registered} images registered — too few for a usable splat")
    log.append(f"pose estimation succeeded ({registered} images registered)")
