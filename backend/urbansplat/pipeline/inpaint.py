"""Stage 1.7 — video inpainting of dynamic regions (ProPainter).

The mask stage flags people/vehicles. Rather than just excluding them (which leaves
unsupervised holes that fill with noise), we inpaint those regions with temporally
consistent background using ProPainter, processing each yaw direction as its own video
track. The filled frames replace the originals and the masks are cleared, so pose and
training run on a clean, person-free, fully-supervised scene.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..config import settings
from .base import PipelineContext, run_command


def _track_key(stem: str) -> str:
    """A temporal track = one clip + one yaw direction (e.g. s0_v02)."""
    s = re.search(r"s(\d+)_", stem)
    v = re.search(r"_v(\d+)", stem)
    return f"s{s.group(1) if s else '0'}_v{v.group(1) if v else '0'}"


def generate_inpaint(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run or not settings.inpaint_enabled:
        log.append("[inpaint] skipped (dry-run or disabled)")
        return
    if not any(ctx.masks_dir.glob("*.png")):
        log.append("[inpaint] no masks — nothing to inpaint")
        return

    import cv2
    import numpy as np

    # Group perspective views into per-yaw temporal tracks (a coherent walk video each).
    tracks: dict[str, list[Path]] = {}
    for f in sorted(ctx.frames_dir.glob("*.jpg")):
        tracks.setdefault(_track_key(f.stem), []).append(f)

    sz = settings.inpaint_proc_size
    work = ctx.work / "inpaint"
    total = 0
    for key, files in tracks.items():
        files = sorted(files)
        tdir = work / key
        (tdir / "frames").mkdir(parents=True, exist_ok=True)
        (tdir / "masks").mkdir(parents=True, exist_ok=True)

        has_dynamic = False
        for i, f in enumerate(files):
            cv2.imwrite(str(tdir / "frames" / f"{i:05d}.jpg"), cv2.imread(str(f)))
            keep = cv2.imread(str(ctx.masks_dir / f"{f.name}.png"), cv2.IMREAD_GRAYSCALE)
            inv = np.where(keep == 0, 255, 0).astype(np.uint8)   # 255 = region to inpaint
            if inv.any():
                has_dynamic = True
            cv2.imwrite(str(tdir / "masks" / f"{i:05d}.png"), inv)

        if not has_dynamic:
            continue

        out = tdir / "out"
        run_command(
            ["python", "inference_propainter.py",
             "-i", str(tdir / "frames"), "-m", str(tdir / "masks"),
             "-o", str(out), "--save_frames", "--fp16",
             "--width", str(sz), "--height", str(sz),
             "--subvideo_length", str(settings.inpaint_subvideo)],
            log, cwd=settings.propainter_dir,
        )

        # Locate ProPainter's output frame folder (it nests under the input dir name).
        produced = sorted(out.rglob("[0-9]*.png"))
        by_idx = {int(p.stem): p for p in produced}

        # Composite the fill back at full resolution — only inside the masked region,
        # so non-masked pixels keep their original 1440px sharpness.
        for i, f in enumerate(files):
            ip = by_idx.get(i)
            if ip is None:
                continue
            orig = cv2.imread(str(f))
            h, w = orig.shape[:2]
            filled = cv2.resize(cv2.imread(str(ip)), (w, h))
            keep = cv2.imread(str(ctx.masks_dir / f"{f.name}.png"), cv2.IMREAD_GRAYSCALE)
            dyn = keep == 0
            orig[dyn] = filled[dyn]
            cv2.imwrite(str(f), orig)
            total += 1

    # Masks consumed — drop them so pose/train use the clean inpainted frames in full.
    for m in ctx.masks_dir.glob("*.png"):
        m.unlink()

    ctx.metrics["inpainted_frames"] = total
    log.append(f"[inpaint] filled dynamic regions in {total} frames ({len(tracks)} tracks)")
