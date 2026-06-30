# urbansplat — Roadmap

> Positioning: **not** a satellite city-twin (that's Vizzio's game — capital + patents).
> urbansplat is the **open, street-level layer** Vizzio can't do: photorealistic
> ground-level navigable capture. Complement to aerial/satellite, not a clone.

Current state: 360 video → perspective reprojection → COLMAP (direct) → splatfacto →
SOG → web viewer, all GPU, self-hosted. Works end-to-end. Weakness = single-scene
quality (floaters from dynamic objects, thin coverage) and it's a splat, not yet a map.

---

## Phase 1 — Dynamic-object masking (DO FIRST, highest ROI)

**Problem:** people/cars become floaters/"bercak pecah" and corrupt SfM matches.
**Goal:** mask moving objects so they're ignored in both pose and training.

### Approach
1. **Segmentation per perspective view.** GPU model on the worker. Classes to mask:
   person, bicycle, car, motorcycle, bus, truck (+ operator body if visible).
   - Option A: `ultralytics` YOLO11-seg (COCO classes) — fast, trivial to install.
   - Option B: HuggingFace Mask2Former/SegFormer (Cityscapes) — street-tuned, cleaner.
   - Start with A; dilate masks ~10px to cover edges/shadows.
2. **New `mask` stage** (after extract): for each image in `frames/`, write a binary
   mask PNG to `masks/` (0 = ignore dynamic, 255 = keep).
3. **Pose:** pass masks to COLMAP — `feature_extractor --ImageReader.mask_path masks/`
   (COLMAP ignores keypoints where mask=0 → cleaner, more stable poses).
4. **Training:** propagate masks into the nerfstudio dataset. After
   `ns-process-data --skip-colmap`, inject `mask_path` per frame in transforms.json and
   copy masks into the processed dataset (nerfstudio drops masked pixels from the loss →
   no gaussians fit to people/cars).

### Deliverables
- `pipeline/mask.py` + masks wired into pose & train.
- Config: model choice, classes, dilation, on/off toggle.
- Re-run Shinjuku: expect far fewer floaters, cleaner facades.

### Risk
- ns-process-data may not carry masks → post-process transforms.json (manageable).
- Seg adds ~1–3 s/image on GPU (fine).

---

## Phase 2 — Georeferencing (turns splats into a *map*)

**Goal:** anchor each scene to real-world coords + metric scale. Foundation for
stitching and for matching Vizzio's positional-accuracy claim.

1. Extract GPS/IMU telemetry from source (Insta360 `.insv`/mp4 GPMF, or exiftool).
2. Map frames → GPS by timestamp.
3. Align COLMAP camera centers ↔ GPS ENU via Umeyama (similarity) → metric scale +
   real-world orientation; store origin lat/lon + transform in DB.
4. Viewer/map can place scenes on a real map; enables multi-scene alignment.

---

## Phase 3 — Multi-clip stitching (the real moat: block → district)

Single clip → contiguous area. The actual "navigable 3D **map**".
- Block-partition + merge (VastGaussian / CityGaussian / Hierarchical-3DGS).
- Use Phase-2 geo anchors as the global frame to align independent reconstructions.
- LoD streaming so large areas render in-browser.

---

## Phase 4 — Map UX (deliver the "navigable map" promise)

- Walk/fly camera (current orbit cam suits objects, not streets).
- LoD tile streaming, minimap, jump between scenes, real-map overlay.
- Floater cleanup in-viewer; better tone mapping.

---

## Phase 5 — Productize

- Auth + multi-tenant, capture app + guidance, job dashboard, quotas.
- Compression already done (SOG ~20x). Add CPU floater filter to the pipeline.

---

## GTM
- Open-source: hero demo video, Show HN, r/GaussianSplatting / r/selfhosted.
- Niches Vizzio ignores: AEC / real-estate walk-throughs, heritage, street-level survey,
  gov infrastructure audit. Win on **open + self-host + street-level**.

## Order of execution
**Phase 1 (masking) → Phase 2 (georef)** make every scene look professional and become
real map data. Phase 3 (stitching) is the moat, tackled once single-scene is solid.
