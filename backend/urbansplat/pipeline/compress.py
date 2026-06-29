"""Stage 4 — package the web-ready artifact.

For now this passes the trained .ply straight through: the PlayCanvas viewer loads
gaussian-splat .ply directly. SOG/SOGS compression (~95% smaller) is a planned
follow-up via PlayCanvas `splat-transform`; wire it here when ready.
"""

from __future__ import annotations

from ..config import settings
from .base import PipelineContext, StageError


def compress(ctx: PipelineContext, log: list[str]) -> None:
    if not ctx.splat_ply.exists():
        raise StageError("no trained .ply to package")

    ctx.output = ctx.splat_ply
    ctx.output_format = "ply"
    size = ctx.splat_ply.stat().st_size
    ctx.metrics["compressed_bytes"] = size

    if settings.dry_run:
        log.append("[dry-run] passthrough splat.ply")
    else:
        log.append(f"packaged splat.ply ({size} bytes)")
