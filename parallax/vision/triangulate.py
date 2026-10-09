"""Linear (DLT) triangulation from two calibrated views."""

from __future__ import annotations

import numpy as np


def _design(p1: np.ndarray, p2: np.ndarray, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Stack the four DLT constraints per point -> (N, 4, 4)."""
    p1, p2 = np.atleast_2d(p1), np.atleast_2d(p2)
    rows = [
        p1[:, 0:1] * P1[2] - P1[0],
        p1[:, 1:2] * P1[2] - P1[1],
        p2[:, 0:1] * P2[2] - P2[0],
        p2[:, 1:2] * P2[2] - P2[1],
    ]
    return np.stack(rows, axis=1)


def triangulate_many(
    p1: np.ndarray, p2: np.ndarray, P1: np.ndarray, P2: np.ndarray, refine: int = 1
) -> np.ndarray:
    """Triangulate N correspondences. p1, p2: (N, 2) pixels; P1, P2: (3, 4).

    The first pass is the plain algebraic DLT. Each refinement pass re-weights the
    constraints by the inverse projective depth, which turns the algebraic error into
    (approximately) the reprojection error and removes the bias towards the nearer camera.
    """
    A = _design(p1, p2, P1, P2)
    X = _solve(A)
    for _ in range(refine):
        h = np.c_[X, np.ones(len(X))]
        w1 = 1.0 / np.abs(h @ P1[2])
        w2 = 1.0 / np.abs(h @ P2[2])
        weights = np.stack([w1, w1, w2, w2], axis=1)[:, :, None]
        X = _solve(A * weights)
    return X


def _solve(A: np.ndarray) -> np.ndarray:
    _, _, vt = np.linalg.svd(A)
    h = vt[:, -1, :]
    return h[:, :3] / h[:, 3:4]


def triangulate(p1, p2, P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Triangulate a single correspondence -> xyz (3,)."""
    return triangulate_many(np.asarray(p1)[None], np.asarray(p2)[None], P1, P2)[0]


def reprojection_error(
    X: np.ndarray, p1: np.ndarray, p2: np.ndarray, P1: np.ndarray, P2: np.ndarray
) -> np.ndarray:
    """Worst-of-two-views reprojection distance in pixels, shape (N,)."""
    h = np.c_[X, np.ones(len(X))]
    errs = []
    for P, p in ((P1, p1), (P2, p2)):
        proj = h @ P.T
        errs.append(np.linalg.norm(proj[:, :2] / proj[:, 2:3] - p, axis=1))
    return np.maximum(errs[0], errs[1])
