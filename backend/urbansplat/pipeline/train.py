"""Stage 3 — 3D Gaussian Splatting training (nerfstudio splatfacto).

splatfacto runs on gsplat under the hood. nerfstudio also serves a live web viewer while
training; with --viewer.make-share-url it publishes a public URL, which we capture from the
training output and store on the job so the user can watch the splat form in real time.
"""

from __future__ import annotations

import json
import re
import subprocess

from ..config import settings
from ..db import session_scope
from ..models import StageRun
from .base import PipelineContext, StageError, run_command

# nerfstudio/viser prints a share URL once the live viewer is up.
_URL_RE = re.compile(r"https?://[^\s'\"]*(?:viser|nerf\.studio|share)[^\s'\"]*")


def _publish_live_url(job_id: str, url: str) -> None:
    """Persist the live training-viewer URL on the train stage so the API exposes it."""
    try:
        with session_scope() as session:
            row = (session.query(StageRun)
                   .filter(StageRun.job_id == job_id, StageRun.name == "train").one())
            row.metrics = json.dumps({"viewer_url": url, "live": True})
    except Exception:
        pass


def _run_train(cmd: list[str], ctx: PipelineContext, log: list[str]) -> None:
    """Run ns-train, streaming output so the live viewer URL can be surfaced immediately."""
    log.append("$ " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    found = False
    assert proc.stdout is not None
    for line in proc.stdout:
        if "it/s" in line or "s/it" in line:      # skip progress-bar spam
            continue
        log.append(line.rstrip())
        if not found:
            m = _URL_RE.search(line)
            if m:
                url = m.group(0)
                ctx.metrics["viewer_url"] = url
                _publish_live_url(ctx.job_id, url)
                log.append(f"[live] training viewer: {url}")
                found = True
    proc.wait()
    if proc.returncode != 0:
        raise StageError(f"ns-train failed (exit {proc.returncode})")


def train_splat(ctx: PipelineContext, log: list[str]) -> None:
    if settings.dry_run:
        header = (
            "ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            "property float x\nproperty float y\nproperty float z\nend_header\n"
        )
        ctx.splat_ply.write_bytes(header.encode())
        ctx.metrics["num_gaussians"] = 1_000_000
        ctx.metrics["iterations"] = settings.train_iterations
        log.append("[dry-run] wrote stub splat.ply (1M gaussians simulated)")
        return

    train_out = ctx.work / "train"
    # Regularisation flags fight the needle/floater artifacts on sparse street captures.
    _run_train(
        [
            "ns-train", "splatfacto",
            "--data", str(ctx.processed_dir),
            "--output-dir", str(train_out),
            "--experiment-name", "job",
            "--timestamp", "run",
            "--max-num-iterations", str(settings.train_iterations),
            "--pipeline.model.rasterize-mode", "antialiased",
            "--pipeline.model.use-scale-regularization", "True",
            "--pipeline.model.max-gauss-ratio", "5.0",
            "--pipeline.model.cull-alpha-thresh", "0.15",
            "--viewer.make-share-url", "True",
            "--viewer.quit-on-train-completion", "True",
        ],
        ctx, log,
    )

    config = train_out / "job" / "splatfacto" / "run" / "config.yml"
    if not config.exists():
        raise StageError("training did not produce a config — splatfacto run failed")

    # Export the trained gaussians to a .ply the web viewer can load.
    export_dir = ctx.work / "export"
    export_dir.mkdir(parents=True, exist_ok=True)
    run_command(
        ["ns-export", "gaussian-splat", "--load-config", str(config),
         "--output-dir", str(export_dir)],
        log,
    )

    produced = next(export_dir.rglob("*.ply"), None)
    if produced is None:
        raise StageError("export produced no .ply")
    produced.replace(ctx.splat_ply)
    ctx.metrics.pop("viewer_url", None)   # training done — live viewer is gone
    ctx.metrics["iterations"] = settings.train_iterations
    log.append(f"training + export complete → {ctx.splat_ply.name}")
