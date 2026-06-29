"""Job orchestration.

A job runs its four stages sequentially inside one Celery task because the stages
share a job-local working directory on disk. Each stage gets its own StageRun row so
status and logs are inspectable per stage, and a failure isolates to the stage that
broke (the MVP §8 "report SfM failure clearly" requirement).
"""

from __future__ import annotations

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .. import storage
from ..config import settings
from ..db import session_scope
from ..models import Job, JobStatus, Scene, StageRun, StageStatus
from ..pipeline import (
    PipelineContext,
    StageError,
    compress,
    estimate_poses,
    extract_frames,
    train_splat,
)
from .celery_app import celery_app

# Stage name → callable. Order matters; mirrors models.STAGES.
STAGE_FUNCS = {
    "extract": extract_frames,
    "pose": estimate_poses,
    "train": train_splat,
    "compress": compress,
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _set_stage(session, job_id: str, name: str, **fields) -> None:
    row = (
        session.query(StageRun)
        .filter(StageRun.job_id == job_id, StageRun.name == name)
        .one()
    )
    for k, v in fields.items():
        setattr(row, k, v)
    session.commit()


@celery_app.task(name="urbansplat.process_job", bind=True)
def process_job(self, job_id: str) -> str:
    work = Path(tempfile.mkdtemp(prefix=f"job_{job_id}_", dir=_ensure_workdir()))

    # Mark job running, download source.
    with session_scope() as session:
        job = session.get(Job, job_id)
        if job is None:
            return "missing"
        job.status = JobStatus.running
        source_key = job.source_key

    source_video = work / "source.mp4"
    storage.get_file(source_key, source_video)

    ctx = PipelineContext(job_id=job_id, work=work, source_video=source_video)

    try:
        for name in STAGE_FUNCS:
            _run_stage(job_id, name, ctx)
        _finalize_success(job_id, ctx)
        return "succeeded"
    except StageError as exc:
        _finalize_failure(job_id, str(exc))
        return "failed"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _run_stage(job_id: str, name: str, ctx: PipelineContext) -> None:
    log: list[str] = []
    with session_scope() as session:
        _set_stage(session, job_id, name, status=StageStatus.running, started_at=_now())
    try:
        STAGE_FUNCS[name](ctx, log)
    except StageError:
        with session_scope() as session:
            _set_stage(
                session, job_id, name,
                status=StageStatus.failed, finished_at=_now(), log="\n".join(log),
            )
        raise
    with session_scope() as session:
        _set_stage(
            session, job_id, name,
            status=StageStatus.succeeded,
            finished_at=_now(),
            log="\n".join(log),
            metrics=json.dumps(ctx.metrics),
        )


def _finalize_success(job_id: str, ctx: PipelineContext) -> None:
    splat_key = f"scenes/{job_id}/splat.{ctx.output_format}"
    storage.put_file(splat_key, ctx.output, content_type="application/octet-stream")
    with session_scope() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.succeeded
        session.add(
            Scene(
                job_id=job_id,
                splat_key=splat_key,
                splat_format=ctx.output_format,
                num_gaussians=ctx.metrics.get("num_gaussians"),
                size_bytes=ctx.metrics.get("compressed_bytes"),
            )
        )


def _finalize_failure(job_id: str, error: str) -> None:
    with session_scope() as session:
        job = session.get(Job, job_id)
        job.status = JobStatus.failed
        job.error = error


def _ensure_workdir() -> str:
    Path(settings.work_dir).mkdir(parents=True, exist_ok=True)
    return settings.work_dir
