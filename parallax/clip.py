"""Clip container: two synchronised camera streams plus calibration, stored as one `.npz`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

import numpy as np

from parallax.camera import Camera


@dataclass
class Clip:
    frames: np.ndarray  # (2, N, H, W, 3) uint8, BGR
    cameras: tuple[Camera, Camera]
    sport: str
    fps: float = 240.0


def save_clip(target: str | BinaryIO, clip: Clip) -> None:
    np.savez_compressed(
        target,
        frames=np.asarray(clip.frames, dtype=np.uint8),
        K=np.stack([c.K for c in clip.cameras]),
        R=np.stack([c.R for c in clip.cameras]),
        t=np.stack([c.t for c in clip.cameras]),
        sport=np.array(clip.sport),
        fps=np.array(clip.fps),
    )


def load_clip(source: str | BinaryIO) -> Clip:
    with np.load(source, allow_pickle=False) as data:
        frames = data["frames"]
        height, width = frames.shape[2:4]
        cameras = tuple(
            Camera.from_arrays(data["K"][i], data["R"][i], data["t"][i], width, height)
            for i in range(2)
        )
        return Clip(frames, cameras, str(data["sport"]), float(data["fps"]))
