"""Calibrated pinhole camera.

World frame: x across the playing area, y along it, z up (metres).
Camera frame follows the OpenCV convention: x right, y down, z forward.
A pixel coordinate (u, v) refers to the *centre* of pixel (column u, row v).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, eq=False)
class Camera:
    K: np.ndarray  # (3, 3) intrinsics
    R: np.ndarray  # (3, 3) world -> camera rotation
    t: np.ndarray  # (3,)   world -> camera translation
    width: int
    height: int

    @classmethod
    def look_at(
        cls,
        position,
        target,
        focal_px: float,
        size: tuple[int, int],
        up=(0.0, 0.0, 1.0),
    ) -> Camera:
        width, height = size
        pos = np.asarray(position, dtype=float)
        forward = np.asarray(target, dtype=float) - pos
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, np.asarray(up, dtype=float))
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        R = np.stack([right, down, forward])
        K = np.array(
            [[focal_px, 0.0, (width - 1) / 2], [0.0, focal_px, (height - 1) / 2], [0.0, 0.0, 1.0]]
        )
        return cls(K=K, R=R, t=-R @ pos, width=width, height=height)

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix."""
        return self.K @ np.hstack([self.R, self.t[:, None]])

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def focal(self) -> float:
        return float(self.K[0, 0])

    def depth(self, X: np.ndarray) -> np.ndarray:
        """Distance along the optical axis."""
        return (np.asarray(X, dtype=float) @ self.R.T + self.t)[..., 2]

    def project(self, X: np.ndarray) -> np.ndarray:
        """World points (..., 3) -> pixel coordinates (..., 2)."""
        cam = np.asarray(X, dtype=float) @ self.R.T + self.t
        pix = cam @ self.K.T
        return pix[..., :2] / pix[..., 2:3]

    def in_frame(self, X: np.ndarray, margin: float = 0.0) -> np.ndarray:
        uv = self.project(X)
        ok = self.depth(X) > 0
        ok &= (uv[..., 0] >= margin) & (uv[..., 0] <= self.width - 1 - margin)
        ok &= (uv[..., 1] >= margin) & (uv[..., 1] <= self.height - 1 - margin)
        return ok

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        return self.K, self.R, self.t, self.width, self.height

    @classmethod
    def from_arrays(cls, K, R, t, width, height) -> Camera:
        return cls(
            K=np.asarray(K, dtype=float),
            R=np.asarray(R, dtype=float),
            t=np.asarray(t, dtype=float),
            width=int(width),
            height=int(height),
        )
