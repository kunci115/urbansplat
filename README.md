<h1 align="center">urbansplat</h1>

<p align="center"><b>Turn 360° video into a navigable 3D map. Open source. Self-hostable. Your data stays yours.</b></p>

<p align="center">
  An open-source alternative to Matterport / Luma / Vizzio for street-level capture —
  powered by 3D Gaussian Splatting.
</p>

---

> ⚠️ **Status: early MVP scaffold.** End-to-end orchestration works today in **dry-run**
> (no GPU needed). Real reconstruction needs COLMAP + gsplat on a CUDA host — see
> [Real reconstruction](#real-reconstruction).

## What it does

Upload a single 360° video clip → urbansplat extracts frames, estimates camera poses,
trains a Gaussian Splat, compresses it for the web, and serves a navigable 3D scene.
Async, self-hosted, one command to run.

```
360° mp4 ─► extract ─► pose (COLMAP) ─► train (gsplat) ─► compress (SOG) ─► web viewer
```

## Quickstart

```bash
git clone https://github.com/kunci115/urbansplat.git
cd urbansplat
cp .env.example .env
docker compose up --build
```

Then open:

| Service | URL |
|---|---|
| Viewer / dashboard | http://localhost:8080 |
| API docs (Swagger) | http://localhost:8000/docs |
| MinIO console | http://localhost:9001 (urbansplat / urbansplat) |

Upload a video in the dashboard. Watch the four stage pills go green. Click the scene to view.

In **dry-run** (default) every stage produces a stub artifact, so you can exercise the
whole upload → queue → status → scene flow with no GPU and no COLMAP install.

## Architecture

```
HTTP ─► FastAPI (api)  ──► Postgres (jobs / stages / scenes)
           │  upload ──► MinIO (source media)
           └─ enqueue ─► Redis ─► Celery worker
                                    └─ pipeline stages (sequential, per-job scratch dir)
                                         extract · pose · train · compress
                                    artifacts ─► MinIO ─► static viewer (SuperSplat)
```

- **Stage isolation** — each stage has its own DB row, logs, and metrics. Failures
  (especially pose estimation) report clearly instead of emitting silent garbage.
- **GPU tools shelled out** — COLMAP / gsplat invoked via subprocess so they can be
  swapped, and skipped entirely in dry-run.

| Layer | Tech | License |
|---|---|---|
| API | FastAPI | MIT |
| Queue | Celery + Redis | BSD / MIT |
| DB | PostgreSQL | PostgreSQL |
| Storage | MinIO (S3) | AGPLv3 ⚠ |
| Pose | COLMAP `panorama_sfm` | BSD |
| Trainer | gsplat | Apache-2.0 |
| Compression | SOG (PlayCanvas) | open |
| Viewer | SuperSplat | MIT |

## Real reconstruction

Dry-run is for wiring/dev. For actual splats:

1. Run the **worker** on an NVIDIA GPU host (RTX 3090/4090 fine; ≥12GB VRAM).
2. Install [COLMAP](https://colmap.github.io/) (with `panorama_sfm`) and
   [gsplat](https://github.com/nerfstudio-project/gsplat) on that host / image.
3. Set `URBANSPLAT_DRY_RUN=0` in `.env`.
4. Enable the `nvidia` device reservation in `docker-compose.yml` (commented block).

Typical single-scene time: ~30–40 min train on a 4090 (gsplat ~8 min) **plus** COLMAP
pose time, which often dominates for 360 input.

## Capture guidance (matters a lot)

360 urban footage is the hardest case for pose estimation. To get usable results:

- **Lock exposure.** Auto-exposure drift destabilises training.
- **Walk slowly, steady.** Motion blur poisons SfM — blurry frames are auto-dropped.
- **Clean the lens.** A smudge smears across a huge part of the panorama.
- **Avoid empty textureless walls / heavy glass** where possible.

## Scope

This is the **single-clip MVP**. Out of scope for now: multi-segment city stitching,
real-time processing, measurement tools, auth/billing. See
[docs/MVP_SCOPE.md](docs/MVP_SCOPE.md).

## License

MIT (intended). Note MinIO is AGPLv3 — swap for SeaweedFS/Garage if that matters to you.
