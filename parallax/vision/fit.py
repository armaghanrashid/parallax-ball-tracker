"""Piecewise-parabola trajectory model, split at the bounce."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

MIN_SEGMENT_POINTS = 4
MIN_POINTS = 6
BOUNCE_ZONE_M = 0.15  # lowest sample must be this close to the ground to suspect a bounce
ROBUST_SCALE_M = 0.01  # soft-L1 knee: residuals above ~1 cm are treated as outliers


class InsufficientData(ValueError):
    """Not enough triangulated points to fit a trajectory."""


@dataclass(frozen=True, eq=False)
class Segment:
    """One ballistic arc: position(t) = c0 + c1 * (t - tref) + c2 * (t - tref)^2 per axis."""

    t_start: float
    t_end: float
    tref: float
    coef: np.ndarray  # (3 axes, 3 powers)

    @classmethod
    def ballistic(cls, t_start, t_end, p0, v0, a, tref=None) -> Segment:
        """Build from position p0 and velocity v0 at `tref`, constant acceleration a."""
        coef = np.stack(
            [np.asarray(p0, float), np.asarray(v0, float), 0.5 * np.asarray(a, float)], axis=1
        )
        return cls(t_start, t_end, t_start if tref is None else tref, coef)

    def position(self, t):
        tau = np.asarray(t, dtype=float)[..., None] - self.tref
        c = self.coef
        return c[:, 0] + c[:, 1] * tau + c[:, 2] * tau**2

    def solve_axis(self, axis: int, value: float) -> list[float]:
        """All real times at which the given axis equals `value`."""
        c0, c1, c2 = self.coef[axis]
        c0 = c0 - value
        if abs(c2) < 1e-12:
            return [] if abs(c1) < 1e-12 else [self.tref - c0 / c1]
        disc = c1 * c1 - 4 * c2 * c0
        if disc < 0:
            return []
        root = np.sqrt(disc)
        return sorted(self.tref + (-c1 + s * root) / (2 * c2) for s in (-1.0, 1.0))


@dataclass(frozen=True, eq=False)
class BallisticModel:
    segments: tuple[Segment, ...]
    ground_z: float
    bounce: tuple[float, np.ndarray] | None  # (time, ground contact point with z = 0)

    def _segment_at(self, t: float) -> Segment:
        for seg in self.segments:
            if t <= seg.t_end:
                return seg
        return self.segments[-1]

    def position(self, t: float) -> np.ndarray:
        return self._segment_at(float(t)).position(float(t))

    def bounce_point(self) -> np.ndarray | None:
        return None if self.bounce is None else self.bounce[1]

    def cross_y(self, y: float) -> tuple[float, np.ndarray] | None:
        """First time the ball centre reaches the plane Y = y while moving forward.

        Searches the observed arcs first, then extrapolates the last arc.
        """
        tol = 1e-9
        for seg in self.segments:
            for t in seg.solve_axis(1, y):
                if seg.t_start - tol <= t <= seg.t_end + tol:
                    return t, seg.position(t)
        last = self.segments[-1]
        later = [t for t in last.solve_axis(1, y) if t > last.t_end]
        if later:
            return later[0], last.position(later[0])
        return None


def _design(t: np.ndarray, tref: float) -> np.ndarray:
    tau = t - tref
    return np.stack([np.ones_like(tau), tau, tau**2], axis=1)


def _fit_arc(t: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, float]:
    """Robust (soft-L1) quadratic fit per axis. Returns coef (3, 3) and the robust cost."""
    tref = float(t[0])
    A = _design(t, tref)
    n = len(t)
    x0, *_ = np.linalg.lstsq(A, P, rcond=None)  # (3 powers, 3 axes) ordinary least squares

    jac = np.zeros((3 * n, 9))
    for axis in range(3):
        jac[axis::3, axis * 3 : axis * 3 + 3] = A

    def residual(x: np.ndarray) -> np.ndarray:
        return (A @ x.reshape(3, 3).T - P).ravel()

    res = least_squares(
        residual, x0.T.ravel(), jac=lambda _x: jac, loss="soft_l1", f_scale=ROBUST_SCALE_M
    )
    return res.x.reshape(3, 3), float(res.cost)


def _segment(t: np.ndarray, P: np.ndarray) -> Segment:
    coef, _ = _fit_arc(t, P)
    return Segment(float(t[0]), float(t[-1]), float(t[0]), coef)


def _nearest_root(seg: Segment, value: float, t_near: float) -> float | None:
    roots = seg.solve_axis(2, value)
    return min(roots, key=lambda r: abs(r - t_near)) if roots else None


def _best_split(t: np.ndarray, P: np.ndarray, ground_z: float) -> int | None:
    """Index k so that [:k] is the pre-bounce arc and [k:] the post-bounce arc, or None."""
    n = len(t)
    i = int(np.argmin(P[:, 2]))
    if P[i, 2] > ground_z + BOUNCE_ZONE_M:
        return None
    single_cost = _fit_arc(t, P)[1]
    best_k, best_cost = None, np.inf
    for k in range(max(MIN_SEGMENT_POINTS, i - 1), min(n - MIN_SEGMENT_POINTS, i + 2) + 1):
        cost = _fit_arc(t[:k], P[:k])[1] + _fit_arc(t[k:], P[k:])[1]
        if cost < best_cost:
            best_k, best_cost = k, cost
    if best_k is not None and 2.0 * best_cost < single_cost:
        return best_k
    return None


def fit_trajectory(points: np.ndarray, ground_z: float) -> BallisticModel:
    """Fit a piecewise parabola to triangulated points.

    points: (N, 4) array of [t, x, y, z]. ground_z is the ball-centre height at ground contact
    (the ball radius). A single split is allowed, at the bounce.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < MIN_POINTS:
        raise InsufficientData(f"need at least {MIN_POINTS} points, got {len(pts)}")
    pts = pts[np.argsort(pts[:, 0])]
    t, P = pts[:, 0], pts[:, 1:]

    k = _best_split(t, P, ground_z)
    if k is None:
        return BallisticModel((_segment(t, P),), ground_z, None)

    pre, post = _segment(t[:k], P[:k]), _segment(t[k:], P[k:])
    t_mid = 0.5 * (t[k - 1] + t[k])
    t_pre = _nearest_root(pre, ground_z, t_mid)
    t_post = _nearest_root(post, ground_z, t_mid)
    if t_pre is None and t_post is None:
        tb, xy = t_mid, pre.position(t_mid)[:2]
    else:
        roots = [r for r in (t_pre, t_post) if r is not None]
        tb = float(np.mean(roots))
        xy = np.mean(
            [seg.position(r)[:2] for seg, r in ((pre, t_pre), (post, t_post)) if r is not None],
            axis=0,
        )
    pre = Segment(pre.t_start, tb, pre.tref, pre.coef)
    post = Segment(tb, post.t_end, post.tref, post.coef)
    return BallisticModel((pre, post), ground_z, (tb, np.array([xy[0], xy[1], 0.0])))
