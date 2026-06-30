"""Georeferencing (experimental) — anchor a reconstruction to the real world.

If the source video carries GPS telemetry (e.g. Insta360/GoPro), we align the COLMAP
camera track to a local ENU (east/north/up, metres) frame via a Umeyama similarity fit.
That yields a real-world origin (lat/lon) and a metric scale for the splat, stored in
the stage metrics. Without GPS the scene is simply left un-georeferenced.

Folded into the pose stage so it needs no extra DB stage/column.
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

from .base import PipelineContext


def _extract_gps(video: Path) -> list[tuple[float, float, float]]:
    """Pull an ordered GPS track (lat, lon, alt) from embedded telemetry via exiftool."""
    try:
        r = subprocess.run(
            ["exiftool", "-ee", "-n", "-json", "-GPSLatitude", "-GPSLongitude",
             "-GPSAltitude", str(video)],
            capture_output=True, text=True,
        )
        data = json.loads(r.stdout or "[]")
    except Exception:
        return []
    track: list[tuple[float, float, float]] = []
    for entry in data:
        lat, lon = entry.get("GPSLatitude"), entry.get("GPSLongitude")
        if lat is None or lon is None:
            continue
        track.append((float(lat), float(lon), float(entry.get("GPSAltitude") or 0.0)))
    return track


def _to_enu(track: list[tuple[float, float, float]]):
    """Convert lat/lon/alt to local ENU metres relative to the first fix."""
    import numpy as np

    lat0, lon0, alt0 = track[0]
    R = 6378137.0
    out = []
    for lat, lon, alt in track:
        e = math.radians(lon - lon0) * R * math.cos(math.radians(lat0))
        n = math.radians(lat - lat0) * R
        out.append((e, n, alt - alt0))
    return np.array(out, dtype=float)


def _umeyama(src, dst):
    """Similarity transform (scale s, rotation R, translation t) mapping src→dst."""
    import numpy as np

    src_mean = src.mean(0)
    dst_mean = dst.mean(0)
    sc = src - src_mean
    dc = dst - dst_mean
    cov = (dc.T @ sc) / src.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var = (sc ** 2).sum() / src.shape[0]
    s = (D * np.diag(S)).sum() / var if var > 0 else 1.0
    t = dst_mean - s * R @ src_mean
    rmse = float(np.sqrt((((s * (R @ src.T).T + t) - dst) ** 2).sum(1)).mean())
    return float(s), R, t, rmse


def _cam_centers(transforms: Path):
    """Per-frame COLMAP camera centres + a normalised capture time (from raw index)."""
    import numpy as np

    data = json.loads(transforms.read_text())
    frames = data.get("frames", [])
    centers, times = [], []
    for f in frames:
        m = np.array(f["transform_matrix"], dtype=float)
        centers.append(m[:3, 3])
        digits = "".join(c for c in Path(f["file_path"]).stem if c.isdigit())
        times.append(int(digits[:5]) if digits else 0)
    if not centers:
        return np.empty((0, 3)), np.empty((0,))
    t = np.array(times, dtype=float)
    t = (t - t.min()) / (t.ptp() or 1.0)   # normalise to 0..1
    return np.array(centers), t


def georeference(ctx: PipelineContext, log: list[str]) -> None:
    track = _extract_gps(ctx.source_video)
    if len(track) < 3:
        log.append("[geo] no/insufficient GPS telemetry — scene not georeferenced")
        return
    import numpy as np

    enu = _to_enu(track)
    centers, ctime = _cam_centers(ctx.processed_dir / "transforms.json")
    if len(centers) < 3:
        log.append("[geo] too few registered cameras to georeference")
        return

    # Match each camera to a GPS fix by proportional capture time.
    gt = np.linspace(0.0, 1.0, len(enu))
    enu_at = np.stack([np.interp(ctime, gt, enu[:, k]) for k in range(3)], axis=1)
    s, R, t, rmse = _umeyama(centers, enu_at)

    lat0, lon0, alt0 = track[0]
    ctx.metrics["geo"] = {
        "origin_lat": lat0, "origin_lon": lon0, "origin_alt": alt0,
        "scale_m_per_unit": round(s, 6), "rotation": R.tolist(),
        "translation": t.tolist(), "rmse_m": round(rmse, 3),
        "gps_points": len(track), "matched_cams": int(len(centers)),
    }
    log.append(f"[geo] georeferenced @ ({lat0:.6f},{lon0:.6f}); "
               f"scale {s:.3f} m/unit, rmse {rmse:.2f} m")
