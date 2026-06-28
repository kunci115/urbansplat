"""Celery application."""

from __future__ import annotations

from celery import Celery

from ..config import settings

celery_app = Celery(
    "urbansplat",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["urbansplat.worker.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,   # GPU jobs are long; one at a time per worker
    result_expires=86400,
)
