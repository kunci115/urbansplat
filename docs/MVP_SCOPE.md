# urbansplat — MVP Scope

> Open-source, self-hostable platform: 360° video → navigable 3D Gaussian Splat map.
> This doc defines the **first shippable version**. No code yet — scope contract.

Status: DRAFT · Date: 2026-06-29

---

## 1. MVP Goal (one sentence)

Upload a single 360° video clip → get back a compressed, web-viewable 3D Gaussian Splat scene, processed async on a GPU, fully self-hosted.

That is it. One clip in, one navigable splat out.

---

## 2. Guiding Principles

1. **Single-segment first.** No multi-clip city stitching in MVP. That is the research frontier (VastGaussian / CityGaussian) and a v2+ problem.
2. **Offline batch, not real-time.** Honest framing: minutes-to-hours per scene.
3. **De-risk pose estimation before anything else.** SfM on 360 urban footage = the make-or-break link. Spike it first (see §8).
4. **Adopt, don't reinvent.** Compression, viewer, trainer all exist. Glue them.
5. **Permissive licenses only.** Verify every dependency. Avoid non-commercial model weights and AGPL where it blocks adopters.

---

## 3. In Scope (MVP)

| Capability | Detail |
|---|---|
| Upload | Single 360° equirectangular video file (mp4), web upload + API |
| Frame extraction | Sample frames at configurable rate, drop blurry frames |
| Pose estimation | COLMAP `panorama_sfm` cubemap path (6 perspective faces/frame, rig-constrained BA) |
| 3DGS training | gsplat (primary) — fast, low VRAM. Configurable iterations |
| Compression | Export to SOG/SOGS compressed format (not raw PLY) |
| Web viewer | SuperSplat-based viewer, loads compressed scene, free navigation |
| Job orchestration | Async queue: submit → poll status → fetch result |
| Storage | Object store for input media + output artifacts |
| Metadata | Job state, scene metadata, params in Postgres |
| Self-host | docker-compose up → working stack on one GPU box |

---

## 4. Out of Scope (explicitly deferred)

- ❌ Multi-segment / multi-clip stitching into one map (v2 — city scale)
- ❌ Aerial + ground fusion (Horizon-GS territory)
- ❌ Real-time / live SLAM splatting
- ❌ Measurement, annotation, floorplan tools (Matterport features)
- ❌ Dynamic object removal (cars/people) — document as known limitation
- ❌ Multi-tenant / auth / billing (single-operator MVP)
- ❌ Mobile capture app — bring-your-own 360 footage
- ❌ Photo-set input — video only in MVP (photos = fast-follow)
- ❌ Auto exposure-drift correction — document capture guidance instead

---

## 5. Tech Stack (locked for MVP)

| Layer | Choice | License | Note |
|---|---|---|---|
| API | FastAPI | MIT | async job submit/status |
| Queue | Celery + Redis | BSD/MIT | GPU job dispatch. Revisit if heavy |
| DB | PostgreSQL (+ PostGIS later) | PostgreSQL | metadata |
| Object store | MinIO (S3 API) | **AGPLv3 ⚠** | self-host. Flag for adopters; SeaweedFS = Apache fallback |
| Pose | COLMAP panorama_sfm | BSD | cubemap route. VGGT/MASt3R as later fallback |
| Trainer | gsplat | Apache-2.0 | primary. Brush = no-CUDA fallback |
| Compression | SOG / SOGS | open | mandatory before serving |
| Viewer | SuperSplat / PlayCanvas | MIT | web |

**License action items (do before depending):**
- [ ] Confirm VGGT / MASt3R / DUSt3R weight licenses (many are CC-BY-NC = non-commercial → can't ship). Keep COLMAP as license-safe default.
- [ ] Decide MinIO (AGPL) vs SeaweedFS/Garage (Apache) for adopter-friendliness.

---

## 6. Pipeline Stages (MVP data flow)

```
upload(mp4 360)
  → extract frames (blur filter, frame-rate sample)
  → COLMAP panorama_sfm (ERP → cubemap 6 faces → match → BA → poses + sparse)
  → gsplat train (N iters, configurable)
  → compress (PLY → SOG/SOGS)
  → store artifacts + register scene
  → viewer loads compressed scene
```

Each stage = a queue task with status + logs. Failure isolates to a stage (SfM fail must report clearly, not silent-garbage).

---

## 7. MVP Milestones

**M0 — Pose spike (BLOCKING, do first).** Standalone: real messy 360 urban clip → COLMAP panorama_sfm → usable poses. Prove or kill. No platform code until this passes. (See §8.)

**M1 — Pipeline CLI.** End-to-end on one machine, no API: `mp4 → frames → poses → gsplat → SOG`. Hardcoded params OK.

**M2 — Orchestration.** Wrap stages in Celery tasks. FastAPI submit/status endpoints. Postgres job state. MinIO artifacts.

**M3 — Viewer.** Web upload form + SuperSplat viewer loading compressed output. End-to-end through browser.

**M4 — Self-host packaging.** docker-compose (api, worker, redis, postgres, minio, viewer). One-command bring-up. README quickstart + hosted demo.

**M5 — Launch.** Hero demo video, Show HN, Reddit (r/GaussianSplatting, r/selfhosted), X. Demo live BEFORE post.

---

## 8. Biggest Risk → De-risk Plan

**Risk:** robust camera-pose estimation from real-world 360 urban footage. Textureless facades, glass, repetitive structure, moving cars/people, fisheye-seam distortion, exposure flicker. If poses fail → all downstream is garbage. Every other component is solved/buyable; this one is not.

**De-risk (M0, before platform code):**
1. Capture/obtain 2-3 real 360 urban clips (varied: open street, tight alley, glass facade).
2. Run COLMAP `panorama_sfm` cubemap path. Measure: registration rate, reprojection error, visual sanity of sparse cloud.
3. If COLMAP weak → test VGGT / MASt3R poses (mind license for shipping).
4. Define capture guidance (lock exposure, slow walk, clean lens) as part of product, not afterthought.
5. **Gate:** if no route gives usable poses on real footage → rescope or stop. Cheaper to learn now.

---

## 9. Definition of Done (MVP)

- [ ] `docker-compose up` → full stack on one GPU box
- [ ] Upload single 360 mp4 via web → async job runs
- [ ] Output = compressed (SOG/SOGS) scene, not raw PLY
- [ ] Navigable in browser viewer at interactive FPS
- [ ] Job status + per-stage logs visible; SfM failure reported clearly
- [ ] README: quickstart (one command) + hero demo video + live demo link
- [ ] All deps license-cleared (no non-commercial weights, MinIO/AGPL decision made)

---

## 10. Known Limitations to Document (not bugs)

- Single clip only — no large-area stitching yet.
- Quality gated by capture discipline (exposure lock, no motion blur, clean lens).
- 360 effective resolution < headline pixel count.
- Dynamic objects (cars/people) may ghost.
- Offline processing: minutes to hours, GPU required (no CPU-only training).
