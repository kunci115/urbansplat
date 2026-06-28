"""Pydantic API schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class StageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    ordinal: int
    status: str
    started_at: datetime | None = None
    finished_at: datetime | None = None
    metrics: str | None = None


class SceneOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    splat_format: str
    num_gaussians: int | None = None
    size_bytes: int | None = None


class JobOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    status: str
    error: str | None = None
    created_at: datetime
    updated_at: datetime
    stages: list[StageOut] = []
    scene: SceneOut | None = None


class JobCreatedOut(BaseModel):
    id: str
    status: str
