"""Stage 2 — camera pose estimation (the make-or-break stage).

We drive COLMAP directly (single shared camera, GPU SIFT, exhaustive matching) because
nerfstudio's `ns-process-data` COLMAP defaults register almost nothing on our 360-derived
perspective views (~2/150), whereas this recipe registers the bulk of them. We then hand
the resulting sparse model to `ns-process-data --skip-colmap`, which only converts it to
the nerfstudio dataset format (transforms.json) the trainer consumes.

Per MVP §8 this is the highest-risk stage; it fails LOUDLY (StageError) rather than
emitting degenerate poses that silently ruin training.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from ..config import settings
from .base import PipelineContext, StageError, run_command


def _registered_count(model_dir: Path) -> int:
    """Number of registered images in a COLMAP model, via model_analyzer."""
    r = subprocess.run(
        ["colmap", "model_analyzer", "--path", str(model_dir)],
        capture_output=True, text=True,
    )
    for line in (r.stdout + r.stderr).splitlines():
        if "Registered images" in line:
            try:
                return int(line.split(":")[-1].strip())
            except ValueError:
                return 0
    return 0


def estimate_poses(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        ctx.processed_dir.mkdir(parents=True, exist_ok=True)
        (ctx.processed_dir / "transforms.json").write_text(json.dumps({"frames": [{}] * 8}))
        ctx.metrics["registered_images"] = 8
        log.append("[dry-run] wrote stub nerfstudio dataset")
        return

    db = ctx.colmap_dir / "database.db"
    sparse = ctx.colmap_dir / "sparse"
    sparse.mkdir(parents=True, exist_ok=True)

    run_command(
        ["colmap", "feature_extractor", "--database_path", str(db),
         "--image_path", str(ctx.frames_dir),
         "--ImageReader.single_camera", "1", "--SiftExtraction.use_gpu", "1"],
        log,
    )
    run_command(
        ["colmap", "exhaustive_matcher", "--database_path", str(db),
         "--SiftMatching.use_gpu", "1"],
        log,
    )
    run_command(
        ["colmap", "mapper", "--database_path", str(db),
         "--image_path", str(ctx.frames_dir), "--output_path", str(sparse)],
        log,
    )

    # COLMAP may emit several disconnected models; keep the largest.
    models = [d for d in sparse.iterdir() if d.is_dir()]
    if not models:
        raise StageError("COLMAP mapper produced no model — pose estimation failed")
    best = max(models, key=_registered_count)
    n = _registered_count(best)
    ctx.metrics["registered_images"] = n
    ctx.metrics["colmap_models"] = len(models)
    log.append(f"COLMAP registered {n} images (largest of {len(models)} model(s))")
    if n < 8:
        raise StageError(
            f"only {n} images registered — too few for a usable splat. "
            "Likely too little camera translation, dynamic scene, or exposure drift."
        )

    # Convert the existing COLMAP model to the nerfstudio dataset format (no re-running
    # COLMAP). --colmap-model-path is relative to --output-dir, so stage the chosen model
    # at the location ns-process-data expects.
    rel_model = Path("colmap") / "sparse" / "0"
    staged = ctx.processed_dir / rel_model
    staged.mkdir(parents=True, exist_ok=True)
    for f in best.glob("*"):
        shutil.copy(f, staged / f.name)
    run_command(
        ["ns-process-data", "images", "--data", str(ctx.frames_dir),
         "--output-dir", str(ctx.processed_dir),
         "--skip-colmap", "--colmap-model-path", str(rel_model)],
        log,
    )
    if not (ctx.processed_dir / "transforms.json").exists():
        raise StageError("failed to convert COLMAP model to nerfstudio format")
    log.append("pose estimation succeeded")
