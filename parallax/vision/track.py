"""Constant-acceleration Kalman tracking with gated nearest-neighbour association."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Discrete white-noise-jerk covariance shape for one axis (position, velocity, acceleration), dt=1.
_Q_AXIS = np.array([[1 / 20, 1 / 8, 1 / 6], [1 / 8, 1 / 3, 1 / 2], [1 / 6, 1 / 2, 1.0]])
_F_AXIS = np.array([[1.0, 1.0, 0.5], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]])


def _blockdiag_axes(m: np.ndarray) -> np.ndarray:
    """Expand a per-axis (pos, vel, acc) matrix to the state order [px, py, vx, vy, ax, ay]."""
    out = np.zeros((6, 6))
    for i in range(3):
        for j in range(3):
            out[2 * i, 2 * j] = m[i, j]
            out[2 * i + 1, 2 * j + 1] = m[i, j]
    return out


F = _blockdiag_axes(_F_AXIS)
Q_UNIT = _blockdiag_axes(_Q_AXIS)
H = np.zeros((2, 6))
H[0, 0] = H[1, 1] = 1.0
MIN_HITS = 3


class Track:
    """A Kalman-filtered image-space track. Time unit is one frame."""

    def __init__(self, first_frame: int, z: np.ndarray, sigma: float, max_speed: float):
        self.x = np.array([z[0], z[1], 0.0, 0.0, 0.0, 0.0])
        v, a = max_speed / 3.0, max_speed / 6.0
        self.P = np.diag([sigma**2, sigma**2, v**2, v**2, a**2, a**2])
        self.first_frame = first_frame
        self.last_frame = first_frame
        self.misses = 0  # consecutive
        self.measurements: dict[int, np.ndarray] = {first_frame: np.asarray(z, float)}
        self.states: dict[int, np.ndarray] = {first_frame: self.x[:2].copy()}
        self.nis_sum = 0.0

    def predict(self, q: float) -> tuple[np.ndarray, np.ndarray]:
        """Advance one frame. Returns predicted position and innovation covariance (needs R)."""
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + q * Q_UNIT
        return self.x[:2], self.P[:2, :2]

    def copy(self) -> Track:
        clone = Track.__new__(Track)
        clone.__dict__.update(self.__dict__)
        clone.measurements = dict(self.measurements)
        clone.states = dict(self.states)
        return clone

    def update(self, frame: int, z: np.ndarray, r: float) -> float:
        S = H @ self.P @ H.T + r * np.eye(2)
        innovation = z - H @ self.x
        gain = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + gain @ innovation
        self.P = (np.eye(6) - gain @ H) @ self.P
        self.misses = 0
        self.last_frame = frame
        self.measurements[frame] = np.asarray(z, float)
        self.states[frame] = self.x[:2].copy()
        nis = float(innovation @ np.linalg.solve(S, innovation))
        self.nis_sum += nis
        return nis

    def coast(self, frame: int) -> None:
        self.misses += 1
        self.states[frame] = self.x[:2].copy()

    @property
    def hits(self) -> int:
        return len(self.measurements)

    @property
    def speed(self) -> float:
        """Mean image speed over the track's lifetime, in pixels per frame."""
        frames = sorted(self.measurements)
        if len(frames) < 2:
            return 0.0
        path = np.array([self.measurements[f] for f in frames])
        return float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum() / (frames[-1] - frames[0]))

    def absorb(self, other: Track) -> None:
        """Take over the measurements of `other` for frames this track did not measure."""
        for frame, z in other.measurements.items():
            if frame not in self.measurements:
                self.measurements[frame] = z
                self.states.setdefault(frame, z.copy())

    def consistent_with(self, other: Track, max_gap: int = 15) -> bool:
        """Do the two tracks plausibly follow the same object?

        Overlapping tracks must agree on the frames both measured. Disjoint tracks must be
        joined by a short gap that the earlier one's constant-velocity extrapolation bridges.
        """
        common = sorted(self.measurements.keys() & other.measurements.keys())
        if len(common) >= 2:
            gaps = [np.linalg.norm(self.measurements[f] - other.measurements[f]) for f in common]
            return float(np.median(gaps)) <= 1.5
        if common:
            return False
        first, second = (
            (self, other) if min(self.measurements) < min(other.measurements) else (other, self)
        )
        end, start = max(first.measurements), min(second.measurements)
        gap = start - end
        if gap < 1 or gap > max_gap or max(second.measurements) < end:
            return False
        recent = sorted(first.measurements)[-3:]
        pts = np.array([first.measurements[f] for f in recent])
        velocity = (pts[-1] - pts[0]) / max(recent[-1] - recent[0], 1)
        predicted = pts[-1] + velocity * gap
        return bool(np.linalg.norm(predicted - second.measurements[start]) <= 8.0 + 4.0 * gap)

    def measured(self, n_frames: int) -> np.ndarray:
        """(n_frames, 2) associated detections; NaN where the track had none."""
        out = np.full((n_frames, 2), np.nan)
        for f, z in self.measurements.items():
            if f < n_frames:
                out[f] = z
        return out

    def filtered(self, n_frames: int) -> np.ndarray:
        """(n_frames, 2) filter positions (predictions where nothing was measured); NaN
        outside the track's lifetime."""
        out = np.full((n_frames, 2), np.nan)
        for f, p in self.states.items():
            if f < n_frames:
                out[f] = p
        return out


class Tracker:
    """Multi-hypothesis tracker for a single fast ball among clutter.

    Every unexplained detection seeds a candidate track. Each frame, candidates are predicted
    and matched to detections inside a Mahalanobis gate (chi-square, 2 dof). A candidate dies
    after `max_misses` consecutive frames without a match. `best()` returns the longest track
    that moves faster than `min_speed`, which discards static or slowly drifting clutter.
    """

    def __init__(
        self,
        *,
        max_speed: float = 15.0,  # px / frame, bounds the first association
        meas_sigma: float = 0.3,  # px
        process_sigma: float = 2.5,  # px / frame^2.5; large enough to follow a bounce
        gate: float = 13.8,  # chi-square 99.9 % for 2 dof
        max_misses: int = 8,
        min_speed: float = 1.5,
        max_tracks: int = 40,
        max_deviation: float = 4.0,  # px; hard cap on how far a detection may be from the
        deviation_per_miss: float = 5.0,  # prediction, growing with each missed frame
    ):
        self.max_speed = max_speed
        self.r = meas_sigma**2
        self.q = process_sigma**2
        self.sigma = meas_sigma
        self.gate = gate
        self.max_misses = max_misses
        self.min_speed = min_speed
        self.max_tracks = max_tracks
        self.max_dev = max_deviation
        self.dev_per_miss = deviation_per_miss
        self.tracks: list[Track] = []
        self._retired: list[Track] = []
        self.frame = -1

    def update(self, detections: Sequence[tuple[float, float, float]]) -> None:
        """Consume the detections of the next frame."""
        self.frame += 1
        frame = self.frame
        zs = [np.array([d[0], d[1]]) for d in detections]

        pairs = []
        for ti, track in enumerate(self.tracks):
            pos, cov = track.predict(self.q)
            S_inv = np.linalg.inv(cov + self.r * np.eye(2))
            for di, z in enumerate(zs):
                d = z - pos
                dist = float(d @ S_inv @ d)
                cap = self.max_dev + self.dev_per_miss * (track.misses + 1)
                if track.hits < 3:  # velocity not yet known: only the speed bound applies
                    cap = self.max_speed * (track.misses + 1)
                if dist <= self.gate and float(np.hypot(*d)) <= cap:
                    # Confirmed tracks claim detections first, so a freshly seeded duplicate of
                    # an existing track cannot steal the ball from it.
                    pairs.append((track.hits < MIN_HITS + 2, dist, ti, di))

        used_tracks: set[int] = set()
        used_dets: set[int] = set()
        for _, _, ti, di in sorted(pairs):
            if ti in used_tracks or di in used_dets:
                continue
            self.tracks[ti].update(frame, zs[di], self.r)
            used_tracks.add(ti)
            used_dets.add(di)

        for ti, track in enumerate(self.tracks):
            if ti not in used_tracks:
                track.coast(frame)

        alive = []
        for track in self.tracks:
            if track.misses > self.max_misses:
                if track.hits >= MIN_HITS:
                    self._retired.append(track)
            else:
                alive.append(track)
        self.tracks = alive
        for di, z in enumerate(zs):
            if di not in used_dets:
                self.tracks.append(Track(frame, z, self.sigma, self.max_speed))
        if len(self.tracks) > self.max_tracks:
            self.tracks.sort(key=lambda t: (-t.hits, t.misses))
            self.tracks = self.tracks[: self.max_tracks]

    def candidates(self) -> list[Track]:
        """Every track worth considering, with tracks that follow the same object merged.

        Longest first. Use this when another sensor (a second camera) can decide which
        candidate is the ball.
        """
        pool = [t for t in (*self._retired, *self.tracks) if t.hits >= MIN_HITS]
        pool.sort(key=lambda t: (-t.hits, t.nis_sum / max(t.hits - 1, 1)))
        merged: list[Track] = []
        for track in pool:
            for core in merged:
                if core.consistent_with(track):
                    core.absorb(track)
                    break
            else:
                merged.append(track.copy())
        merged.sort(key=lambda t: -t.hits)
        return merged

    def best(self) -> Track | None:
        """The longest candidate that moves faster than `min_speed`, or None.

        Single-camera heuristic: static and slowly drifting clutter is discarded by speed.
        """
        for track in self.candidates():
            if track.speed >= self.min_speed:
                return track
        return None
