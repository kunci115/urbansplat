"""Processing pipeline: 360 video → frames → poses → splat → compressed scene."""

from .base import PipelineContext, StageError, run_command
from .compress import compress
from .extract import extract_frames
from .mask import generate_masks
from .pose import estimate_poses
from .train import train_splat

__all__ = [
    "PipelineContext",
    "StageError",
    "run_command",
    "extract_frames",
    "generate_masks",
    "estimate_poses",
    "train_splat",
    "compress",
]
