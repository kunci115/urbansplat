"""API routes: submit jobs, poll status, fetch scene."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from .. import storage
from ..db import get_session
from ..models import STAGES, Job, JobStatus, Scene, StageRun, StageStatus
from ..schemas import JobCreatedOut, JobOut
from ..worker.tasks import process_job

router = APIRouter(prefix="/api", tags=["jobs"])


@router.post("/jobs", response_model=JobCreatedOut, status_code=201)
def create_job(
    file: UploadFile = File(...),
    name: str = Form("untitled"),
    session: Session = Depends(get_session),
) -> JobCreatedOut:
    """Upload a 360 video and enqueue processing."""
    if not file.filename or not file.filename.lower().endswith((".mp4", ".mov", ".mkv")):
        raise HTTPException(400, "expected a video file (.mp4/.mov/.mkv)")

    job_id = str(uuid.uuid4())
    source_key = f"sources/{job_id}/{file.filename}"
    _put_upload(source_key, file)

    job = Job(id=job_id, name=name, source_key=source_key, status=JobStatus.pending)
    for ordinal, stage in enumerate(STAGES):
        job.stages.append(StageRun(name=stage, ordinal=ordinal, status=StageStatus.pending))
    session.add(job)
    session.commit()

    process_job.delay(job_id)
    return JobCreatedOut(id=job_id, status=job.status.value)


def _put_upload(key: str, file: UploadFile) -> str:
    # UploadFile.size is known after read; stream via temp to avoid loading huge files in RAM.
    import tempfile

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        while chunk := file.file.read(8 * 1024 * 1024):
            tmp.write(chunk)
        tmp_path = tmp.name
    storage.put_file(key, tmp_path, content_type=file.content_type or "video/mp4")
    import os

    os.unlink(tmp_path)
    return key


@router.get("/jobs", response_model=list[JobOut])
def list_jobs(session: Session = Depends(get_session)) -> list[Job]:
    return session.query(Job).order_by(Job.created_at.desc()).limit(100).all()


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, session: Session = Depends(get_session)) -> Job:
    job = session.get(Job, job_id)
    if job is None:
        raise HTTPException(404, "job not found")
    return job


@router.get("/scenes/{job_id}/splat")
def get_splat(job_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    """Stream the splat bytes straight from object storage to the browser.

    Proxying (rather than redirecting to a presigned URL) keeps storage internal —
    the browser only ever talks to the API host, never to `minio:9000`.
    """
    scene = session.query(Scene).filter(Scene.job_id == job_id).one_or_none()
    if scene is None:
        raise HTTPException(404, "scene not ready")
    filename = f"scene.{scene.splat_format}"  # extension lets the viewer pick a loader
    headers = {"Content-Disposition": f'inline; filename="{filename}"'}
    try:
        size = storage.stat(scene.splat_key).size
        headers["Content-Length"] = str(size)
    except Exception:
        pass
    return StreamingResponse(
        storage.stream_object(scene.splat_key),
        media_type="application/octet-stream",
        headers=headers,
    )


@router.get("/scenes/{job_id}/ply")
def get_ply(job_id: str, session: Session = Depends(get_session)) -> StreamingResponse:
    """Download the raw 3DGS .ply (for Unity / DCC import). 404 if not retained."""
    scene = session.query(Scene).filter(Scene.job_id == job_id).one_or_none()
    if scene is None:
        raise HTTPException(404, "scene not ready")
    key = f"scenes/{job_id}/splat.ply"
    try:
        size = storage.stat(key).size
    except Exception:
        raise HTTPException(404, "no .ply retained for this scene")
    headers = {
        "Content-Disposition": f'attachment; filename="{job_id}.ply"',
        "Content-Length": str(size),
    }
    return StreamingResponse(
        storage.stream_object(key), media_type="application/octet-stream", headers=headers
    )
