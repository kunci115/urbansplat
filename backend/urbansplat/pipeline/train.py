"""Stage 3 — 3D Gaussian Splatting training (nerfstudio splatfacto).

splatfacto runs on gsplat under the hood (fast, low VRAM). We pin experiment name /
timestamp so the trained config path is predictable, then export a Gaussian-splat .ply.
"""

from __future__ import annotations

from ..config import settings
from .base import PipelineContext, StageError, run_command


def train_splat(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        ctx.splat_ply.write_bytes(header.encode())
        ctx.metrics["num_gaussians"] = 1_000_000
        ctx.metrics["iterations"] = settings.train_iterations
        log.append("[dry-run] wrote stub splat.ply (1M gaussians simulated)")
        return

    train_out = ctx.work / "train"
    # Fixed names make the produced config path deterministic.
    run_command(
        [
            "ns-train", "splatfacto",
            "--data", str(ctx.processed_dir),
            "--output-dir", str(train_out),
            "--experiment-name", "job",
            "--timestamp", "run",
            "--max-num-iterations", str(settings.train_iterations),
            "--viewer.quit-on-train-completion", "True",
        ],
        log,
    )

    config = train_out / "job" / "splatfacto" / "run" / "config.yml"
    if not config.exists():
        raise StageError("training did not produce a config — splatfacto run failed")

    # Export the trained gaussians to a .ply the web viewer can load.
    export_dir = ctx.work / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "ns-export", "gaussian-splat",
            "--load-config", str(config),
            "--output-dir", str(export_dir),
        ],
        log,
    )

    produced = next(export_dir.rglob("*.ply"), None)
    if produced is None:
        raise StageError("export produced no .ply")
    produced.replace(ctx.splat_ply)
    ctx.metrics["iterations"] = settings.train_iterations
    log.append(f"training + export complete → {ctx.splat_ply.name}")
