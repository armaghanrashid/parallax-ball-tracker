"""End-to-end analysis: two synchronised clips in, a line call out."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from parallax.camera import Camera
from parallax.rules import Decision
from parallax.sports import SPORTS
from parallax.vision.detect import detect
from parallax.vision.fit import BallisticModel, InsufficientData, fit_trajectory
from parallax.vision.track import Track, Tracker
from parallax.vision.triangulate import reprojection_error, triangulate_many

NO_DECISION = "NO_DECISION"
REPROJECTION_GATE_PX = 2.0  # discard triangulations whose two views disagree by more than this


@dataclass
class Analysis:
    decision: Decision
    model: BallisticModel | None
    points: np.ndarray  # (N, 4) [t, x, y, z] triangulated ball centres
    frame_index: np.ndarray  # (N,) frame number of each point
    tracks: tuple[np.ndarray, np.ndarray]  # per camera (n_frames, 2) pixels, NaN if unseen
    n_frames: int
    latency_ms: float


MIN_PAIR_POINTS = 6
MIN_TRACK_LENGTH_M = 0.5  # a ball moves; static clutter triangulates to a fixed point
Z_RANGE_M = (-0.2, 8.0)


def _candidates(frames: Sequence[np.ndarray], detect_kwargs: dict) -> list[Track]:
    tracker = Tracker()
    for frame in frames:
        tracker.update(detect(frame, **detect_kwargs))
    return tracker.candidates()


def _pair_support(a: Track, b: Track, n: int, P1: np.ndarray, P2: np.ndarray):
    """Triangulate two candidate tracks against each other.

    Returns the frame indices and 3D points on which the views agree (reprojection error within
    the gate, height plausible). Two tracks that follow the same ball agree on most frames;
    unrelated clutter does not.
    """
    ma, mb = a.measured(n), b.measured(n)
    index = np.flatnonzero(~(np.isnan(ma[:, 0]) | np.isnan(mb[:, 0])))
    if len(index) == 0:
        return index, np.zeros((0, 3))
    xyz = triangulate_many(ma[index], mb[index], P1, P2)
    ok = reprojection_error(xyz, ma[index], mb[index], P1, P2) <= REPROJECTION_GATE_PX
    ok &= (xyz[:, 2] > Z_RANGE_M[0]) & (xyz[:, 2] < Z_RANGE_M[1])
    return index[ok], xyz[ok]


def _match_tracks(cands0: list[Track], cands1: list[Track], n: int, P1, P2):
    """The camera-0 / camera-1 track pair that best explains one moving 3D object."""
    best = (0, None, None, None)
    for a in cands0:
        for b in cands1:
            index, xyz = _pair_support(a, b, n, P1, P2)
            if len(index) < max(best[0] + 1, MIN_PAIR_POINTS):
                continue
            if np.linalg.norm(xyz.max(axis=0) - xyz.min(axis=0)) < MIN_TRACK_LENGTH_M:
                continue
            best = (len(index), a, b, (index, xyz))
    return best


def analyze(
    frames: Sequence[Sequence[np.ndarray]] | np.ndarray,
    cameras: Sequence[Camera],
    sport: str,
    fps: float = 240.0,
) -> Analysis:
    """Detect, track, triangulate, fit and judge. `frames[c][i]` is frame i of camera c."""
    cfg = SPORTS[sport]
    started = time.perf_counter()
    n = len(frames[0])

    def finish(decision, model=None, points=None, index=None, tracks=None):
        points = np.zeros((0, 4)) if points is None else points
        index = np.zeros(0, int) if index is None else index
        if tracks is None:
            tracks = (np.full((n, 2), np.nan), np.full((n, 2), np.nan))
        latency = (time.perf_counter() - started) * 1000.0
        return Analysis(decision, model, points, index, tracks, n, latency)

    no_call = Decision(NO_DECISION, None, None)
    P1, P2 = cameras[0].P, cameras[1].P
    cands0 = _candidates(frames[0], cfg.detect)
    cands1 = _candidates(frames[1], cfg.detect)
    support, a, b, found = _match_tracks(cands0, cands1, n, P1, P2)
    if a is None:
        return finish(no_call)
    tracks = (a.measured(n), b.measured(n))
    index, xyz = found
    points = np.c_[index / fps, xyz]
    try:
        model, points, index = _fit_pruned(points, index, cfg.ball_radius)
        decision = cfg.judge(model)
    except (InsufficientData, ValueError):
        return finish(no_call, points=points, index=index, tracks=tracks)
    return finish(decision, model, points, index, tracks)


def _fit_pruned(points: np.ndarray, index: np.ndarray, ground_z: float):
    """Fit, drop points the fit rejects (> 3 cm and > 4x the median residual), refit."""
    model = fit_trajectory(points, ground_z)
    resid = np.linalg.norm(
        np.array([model.position(t) for t in points[:, 0]]) - points[:, 1:], axis=1
    )
    keep = resid <= max(0.03, 4.0 * float(np.median(resid)))
    if keep.all() or keep.sum() < 6:
        return model, points, index
    return fit_trajectory(points[keep], ground_z), points[keep], index[keep]
