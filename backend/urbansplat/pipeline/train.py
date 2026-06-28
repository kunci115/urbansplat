"""Stage 3 — 3D Gaussian Splatting training (gsplat).

Trains a splat from the COLMAP sparse model + frames. gsplat chosen over splatfacto:
faster, lower VRAM, Apache-2.0. Outputs a raw .ply (compressed in the next stage).
"""

from __future__ import annotations

from ..config import settings
from .base import PipelineContext, StageError, run_command


def train_splat(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        # Minimal valid-looking PLY header so compress stage has an input.
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "element vertex 0\n"
            "property float x\nproperty float y\nproperty float z\n"
            "end_header\n"
        )
        ctx.splat_ply.write_bytes(header.encode())
        ctx.metrics["num_gaussians"] = 1_000_000
        ctx.metrics["iterations"] = settings.train_iterations
        log.append("[dry-run] wrote stub splat.ply (1M gaussians simulated)")
        return

    # gsplat's simple_trainer consumes a COLMAP dataset directory.
    run_command(
        [
            "python", "-m", "gsplat.simple_trainer", "default",
            "--data-dir", str(ctx.colmap_dir),
            "--data-factor", "1",
            "--max-steps", str(settings.train_iterations),
            "--result-dir", str(ctx.work / "gsplat_out"),
        ],
        log,
    )

    produced = next((ctx.work / "gsplat_out").rglob("*.ply"), None)
    if produced is None:
        raise StageError("training produced no .ply output")
    produced.replace(ctx.splat_ply)
    ctx.metrics["iterations"] = settings.train_iterations
    log.append(f"training complete → {ctx.splat_ply.name}")
