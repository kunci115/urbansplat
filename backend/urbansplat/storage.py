"""Object storage wrapper (MinIO / S3-compatible)."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path

from minio import Minio

from .config import settings


@lru_cache
def client() -> Minio:
    return Minio(
        settings.s3_endpoint,
        access_key=settings.s3_access_key,
        secret_key=settings.s3_secret_key,
        secure=settings.s3_secure,
    )


def ensure_bucket() -> None:
    c = client()
    if not c.bucket_exists(settings.s3_bucket):
        c.make_bucket(settings.s3_bucket)


def put_file(key: str, path: str | Path, content_type: str = "application/octet-stream") -> str:
    client().fput_object(settings.s3_bucket, key, str(path), content_type=content_type)
    return key


def put_bytes(key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    client().put_object(
        settings.s3_bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
    )
    return key


def put_stream(key: str, stream, length: int, content_type: str) -> str:
    client().put_object(settings.s3_bucket, key, stream, length=length, content_type=content_type)
    return key


def get_file(key: str, dest: str | Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    client().fget_object(settings.s3_bucket, key, str(dest))
    return dest


def presigned_url(key: str, expires_seconds: int = 3600) -> str:
    from datetime import timedelta

    return client().presigned_get_object(
        settings.s3_bucket, key, expires=timedelta(seconds=expires_seconds)
    )
