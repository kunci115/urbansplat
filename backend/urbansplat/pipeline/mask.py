"""Stage 1.5 — dynamic-object masking.

Segments people/vehicles in each perspective view (YOLO-seg) and writes a binary mask
(0 = ignore, 255 = keep). COLMAP skips features in masked regions (cleaner poses) and
splatfacto drops masked pixels from the loss (no gaussians fit to people/cars), which
removes the floaters/"broken patches" dynamic objects otherwise leave behind.

Masks are named ``<image-filename>.png`` to match COLMAP's mask convention.
"""

from __future__ import annotations

from ..config import settings
from .base import PipelineContext


def _dynamic_class_ids() -> set[int]:
    return {int(x) for x in settings.seg_classes.split(",") if x.strip()}


def generate_masks(ctx: PipelineContext, log: list[str]) -> None:
    ctx.masks_dir.mkdir(parents=True, exist_ok=True)
    if settings.dry_run or not settings.masking_enabled:
        log.append("[mask] skipped (dry-run or disabled)")
        return

    import cv2
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(settings.seg_model)
    dyn = _dynamic_class_ids()
    kernel = None
    if settings.mask_dilate_px > 0:
        kernel = np.ones((settings.mask_dilate_px, settings.mask_dilate_px), np.uint8)

    images = sorted(ctx.frames_dir.glob("*.jpg"))
    total_masked = 0.0
    for img in images:
        res = model.predict(str(img), conf=settings.seg_conf, verbose=False)[0]
        h, w = res.orig_shape
        keep = np.full((h, w), 255, np.uint8)
        if res.masks is not None:
            classes = res.boxes.cls.cpu().numpy()
            for seg, cls in zip(res.masks.data.cpu().numpy(), classes):
                if int(cls) in dyn:
                    m = cv2.resize((seg > 0.5).astype(np.uint8), (w, h),
                                   interpolation=cv2.INTER_NEAREST)
                    keep[m > 0] = 0
        if kernel is not None:
            grown = cv2.dilate((keep == 0).astype(np.uint8), kernel)
            keep[grown > 0] = 0
        total_masked += float((keep == 0).mean())
        cv2.imwrite(str(ctx.masks_dir / f"{img.name}.png"), keep)

    if images:
        ctx.metrics["avg_masked_fraction"] = round(total_masked / len(images), 3)
    log.append(
        f"[mask] {len(images)} masks, "
        f"avg {ctx.metrics.get('avg_masked_fraction', 0) * 100:.1f}% masked"
    )
