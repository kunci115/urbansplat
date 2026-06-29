"""Shared pipeline primitives: context, errors, subprocess helper."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class StageError(RuntimeError):
    """Raised when a pipeline stage fails. Message is surfaced to the user."""


@dataclass
class PipelineContext:
    """Carries per-job working state between stages.

    Stages read inputs from and write outputs into `work`, a job-local scratch dir.
    Each stage appends to `log` lines, returned to the worker for persistence.
    """

    job_id: str
    work: Path                       # job-local working directory
    source_video: Path               # local path to downloaded source video
    frames_dir: Path = field(init=False)
    processed_dir: Path = field(init=False)   # nerfstudio dataset (poses + images)
    colmap_dir: Path = field(init=False)
    splat_ply: Path = field(init=False)       # raw trainer output
    # Final web-ready artifact + its format, set by the compress stage and read by the
    # worker when uploading. Defaults to the raw ply so a passthrough is valid.
    output: Path = field(init=False)
    output_format: str = field(init=False)
    metrics: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frames_dir = self.work / "frames"
        self.processed_dir = self.work / "processed"
        self.colmap_dir = self.work / "colmap"
        self.splat_ply = self.work / "splat.ply"
        self.output = self.splat_ply
        self.output_format = "ply"
        for d in (self.work, self.frames_dir):
            d.mkdir(parents=True, exist_ok=True)


def run_command(cmd: list[str], log: list[str], cwd: str | Path | None = None) -> None:
    """Run an external tool, streaming combined output into `log`.

    Raises StageError on non-zero exit so the stage fails loudly (never silent garbage).
    """
    log.append(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise StageError(f"tool not found: {cmd[0]} ({exc})") from exc
    if proc.stdout:
        log.append(proc.stdout)
    if proc.stderr:
        log.append(proc.stderr)
    if proc.returncode != 0:
        raise StageError(f"command failed (exit {proc.returncode}): {' '.join(cmd)}")
