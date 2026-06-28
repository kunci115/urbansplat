"""Stage 4 — compression to web-ready format (SOG).

Raw PLY is ~1GB for a 4M-Gaussian scene — unservable. SOG (PlayCanvas, open) encodes
Gaussian attributes into WebP channels for ~95% size reduction, and the SuperSplat
viewer loads it natively. Mandatory before serving to a browser.
"""

from __future__ import annotations

import shutil

from ..config import settings
from .base import PipelineContext, StageError, run_command


def compress(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        # Pretend we compressed 1GB → ~42MB (typical SOG ratio).
        ctx.splat_sog.write_bytes(b"SOG\x00stub")
        ctx.metrics["compressed_bytes"] = 42 * 1024 * 1024
        log.append("[dry-run] wrote stub splat.sog (~42MB simulated)")
        return

    # PlayCanvas `splat-transform` CLI converts PLY → SOG.
    if shutil.which("splat-transform") is None:
        raise StageError("splat-transform (PlayCanvas) not found on PATH")

    run_command(
        ["splat-transform", str(ctx.splat_ply), str(ctx.splat_sog)],
        log,
    )
    if not ctx.splat_sog.exists():
        raise StageError("compression produced no output")
    ctx.metrics["compressed_bytes"] = ctx.splat_sog.stat().st_size
    log.append(f"compressed → {ctx.splat_sog.name} ({ctx.metrics['compressed_bytes']} bytes)")
