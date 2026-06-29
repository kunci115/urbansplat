<h1 align="center">urbansplat</h1>

<p align="center"><b>Turn 360° video into a navigable 3D map. Open source. Self-hostable. Your data stays yours.</b></p>

<p align="center">
  An open-source alternative to Matterport / Luma / Vizzio for street-level capture —
  powered by 3D Gaussian Splatting.
</p>

---

> **Status: working MVP.**
> - ✅ Full self-hosted stack, one command (`docker compose up`).
> - ✅ Dry-run mode (no GPU) exercises the whole upload → queue → stages → scene → viewer flow.
> - ✅ Real GPU reconstruction (nerfstudio **splatfacto** + COLMAP) via a GPU overlay.
> - ✅ Real splats render in-browser (PlayCanvas), streamed from storage.
> - ⚠️ 360 video → splat works *if the camera translates* — see [Capture guidance](#capture-guidance-read-this).
> - ⏳ Roadmap: SOG compression (splats currently served as raw `.ply`), multi-segment stitching.

## What it does

Upload a video clip → urbansplat extracts frames (reprojecting 360 panoramas to
perspective views), estimates camera poses, trains a Gaussian Splat, and serves a
navigable 3D scene. Async, self-hosted, one command to run.

```
video ─► extract (+360→perspective) ─► pose (COLMAP) ─► train (splatfacto) ─► package (.ply) ─► web viewer
```

For **equirectangular 360** input each frame is reprojected into perspective ("pinhole")
views before SfM — COLMAP can't pose raw equirectangular frames directly.

## Quickstart (dry-run, no GPU)

```bash
git clone https://github.com/kunci115/urbansplat.git
cd urbansplat
cp .env.example .env
docker compose up --build
```

| Service | URL |
|---|---|
| Viewer / dashboard | http://localhost:8080 |
| API docs (Swagger) | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 (urbansplat / urbansplat) |

Upload a video in the dashboard, watch the four stage pills go green, click the scene
to view. In **dry-run** (default) each stage emits a stub artifact, so you can exercise
the entire flow with no GPU and no COLMAP install.

## Architecture

```
HTTP ─► FastAPI (api)  ──► Postgres (jobs / stages / scenes)
           │  upload ──► MinIO (source media)
           └─ enqueue ─► Redis ─► Celery worker
                                    └─ pipeline stages (sequential, per-job scratch dir)
                                         extract · pose · train · compress
                                    artifacts ─► MinIO ─► API byte-proxy ─► viewer (PlayCanvas)
```

- **Stage isolation** — each stage has its own DB row, logs, and metrics. Failures
  (especially pose estimation) report clearly instead of emitting silent garbage.
- **GPU tools shelled out** — COLMAP / nerfstudio invoked via subprocess, skipped in dry-run.
- **Splat served via API byte-proxy** — the browser streams from the API, never touches
  the internal object store directly.

| Layer | Tech | License |
|---|---|---|
| API | FastAPI | MIT |
| Queue | Celery + Redis | BSD / MIT |
| DB | PostgreSQL | PostgreSQL |
| Storage | MinIO (S3) | AGPLv3 ⚠ |
| 360 → perspective | py360convert | MIT |
| Pose | COLMAP (via `ns-process-data`) | BSD |
| Trainer | nerfstudio splatfacto (gsplat) | Apache-2.0 |
| Output | `.ply` passthrough (SOG planned) | — |
| Viewer | PlayCanvas engine | MIT |

## Real reconstruction (GPU)

The GPU worker is built on the official nerfstudio image (CUDA + torch + splatfacto +
COLMAP + ffmpeg). On an NVIDIA host (RTX 3090/4090, ≥12 GB VRAM):

```bash
# starts the full stack with a GPU worker (DRY_RUN=0, nvidia runtime)
docker compose -f docker-compose.yml -f docker-compose.override.yml -f docker-compose.gpu.yml up -d
```

> Pass **all three** `-f` files: an explicit `-f` disables auto-loading of
> `docker-compose.override.yml` (which carries host port remaps).

Tune in `docker-compose.gpu.yml`: `URBANSPLAT_TRAIN_ITERATIONS` (7000 preview → 30000
final), `URBANSPLAT_FRAME_SAMPLE_FPS`, and the 360 reprojection knobs
(`URBANSPLAT_VIEWS_PER_FRAME`, `URBANSPLAT_PERSPECTIVE_YAW_OFFSET`).

Typical single scene: a few minutes COLMAP + ~5–10 min splatfacto (7000 iters) on a 4090.

## Capture guidance (READ THIS)

Pose estimation is the make-or-break step, and 360 footage is the hardest case. The
single biggest requirement:

- **The camera must TRANSLATE.** Walk or drive *continuously* — never stand still or
  spin in place. No camera motion = no parallax = SfM cannot reconstruct. (Most free
  online 360 clips are static tripod/VR tours and will not work.)
- **No burned-in overlays/watermarks** — they become false static features.
- **Lock exposure.** Auto-exposure drift destabilises training.
- **Move smoothly.** Motion blur poisons SfM — blurry frames are auto-dropped.
- **Clean the lens**, and avoid empty textureless walls / heavy glass / dense moving crowds.

360 cameras (e.g. Insta360) record dual-fisheye natively; **export to equirectangular
(2:1) mp4** before feeding urbansplat.

## Scope

Single-clip MVP. Out of scope for now: multi-segment city stitching, real-time
processing, measurement tools, auth/billing. See [docs/MVP_SCOPE.md](docs/MVP_SCOPE.md).

## Known limitations

- 360 reconstruction needs translating footage (above); static/tripod 360 fails at SfM.
- Splats served as raw `.ply` (large) — SOG compression is the next optimisation.
- Per-frame 360 views share one optical centre; the pipeline reprojects side-on views to
  rely on cross-frame parallax. Robust full-coverage 360 will want COLMAP rig mode +
  dynamic-object masking.

## License

MIT (intended). Note MinIO is AGPLv3 — swap for SeaweedFS/Garage if that matters to you.
