"""Stage 4 — compress the trained splat to a web-ready format.

Raw .ply is huge (100s of MB) and slow to load. PlayCanvas `splat-transform` packs the
gaussians into SOG (WebP-channel encoding, ~10-20x smaller) which the PlayCanvas viewer
loads natively. Falls back to passthrough .ply if the tool is unavailable.
"""

from __future__ import annotations

import shutil

from ..config import settings
from .base import PipelineContext, StageError, run_command


def compress(ctx: PipelineContext, log: list[str]) -> None:
    if not ctx.splat_ply.exists():
        raise StageError("no trained .ply to package")

    raw_size = ctx.splat_ply.stat().st_size

    if settings.dry_run:
        ctx.output = ctx.splat_ply
        ctx.output_format = "ply"
        ctx.metrics["compressed_bytes"] = raw_size
        log.append("[dry-run] passthrough splat.ply")
        return

    sog = ctx.work / "splat.sog"
    if shutil.which("splat-transform"):
        # -H 0 drops spherical-harmonic bands (also avoids splat-transform's WebGPU SH
        # path, which fails headless) → much smaller; -N strips NaN/Inf gaussians.
        run_command(
            ["splat-transform", str(ctx.splat_ply), "-H", "0", "-N", str(sog)], log
        )

    if sog.exists() and sog.stat().st_size > 0:
        ctx.output = sog
        ctx.output_format = "sog"
        size = sog.stat().st_size
        ctx.metrics["compressed_bytes"] = size
        log.append(f"compressed {raw_size} → {size} bytes SOG ({raw_size/max(size,1):.1f}x)")
    else:
        # Fallback: serve the raw ply so a scene is still produced.
        ctx.output = ctx.splat_ply
        ctx.output_format = "ply"
        ctx.metrics["compressed_bytes"] = raw_size
        log.append("splat-transform unavailable/failed — serving raw .ply")
