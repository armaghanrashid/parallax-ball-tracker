"""Ball flight with gravity, quadratic drag and bounces. Sampled at 240 Hz."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

G = 9.80665
AIR_DENSITY = 1.2
RATE_HZ = 240.0

Vec = tuple[float, float, float]


@dataclass(frozen=True)
class BallSpec:
    radius: float  # m
    mass: float  # kg
    cd: float  # drag coefficient
    restitution: float  # vertical speed retained at a bounce
    retention: float  # horizontal speed retained at a bounce (surface friction)

    @property
    def drag_k(self) -> float:
        """Quadratic drag: a = -k |v| v."""
        return 0.5 * AIR_DENSITY * self.cd * math.pi * self.radius**2 / self.mass


SPECS: dict[str, BallSpec] = {
    "tennis": BallSpec(radius=0.0335, mass=0.057, cd=0.55, restitution=0.75, retention=0.72),
    "cricket": BallSpec(radius=0.0356, mass=0.156, cd=0.45, restitution=0.55, retention=0.85),
    "football": BallSpec(radius=0.11, mass=0.43, cd=0.25, restitution=0.65, retention=0.8),
}


@dataclass(frozen=True)
class Bounce:
    t: float
    point: np.ndarray  # contact point on the ground plane (z = 0)
    v_in: Vec
    v_out: Vec


@dataclass
class Flight:
    t: np.ndarray  # (N,)
    pos: np.ndarray  # (N, 3) ball centre
    vel: np.ndarray  # (N, 3)
    bounces: list[Bounce] = field(default_factory=list)


def _accel(v: Vec, k: float) -> Vec:
    s = k * math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    return (-s * v[0], -s * v[1], -G - s * v[2])


def integrate(
    spec: BallSpec,
    p0: Vec,
    v0: Vec,
    *,
    stop_y: float,
    t_max: float = 3.0,
    substeps: int = 20,
    bounce_fn: Callable[[Vec], Vec] | None = None,
) -> Flight:
    """Integrate until the ball centre reaches `stop_y` (or `t_max`).

    Midpoint integration at 240 Hz x `substeps`. When the centre drops to one radius above the
    ground the step is split at the exact contact instant and the velocity is reflected.
    """
    dt = 1.0 / (RATE_HZ * substeps)
    k, r = spec.drag_k, spec.radius
    x, y, z = (float(c) for c in p0)
    v: Vec = (float(v0[0]), float(v0[1]), float(v0[2]))
    reflect = bounce_fn or (
        lambda u: (spec.retention * u[0], spec.retention * u[1], -spec.restitution * u[2])
    )

    times, pos, vel, bounces = [0.0], [(x, y, z)], [v], []
    n_steps = int(t_max * RATE_HZ * substeps)
    for step in range(1, n_steps + 1):
        t_prev = (step - 1) * dt
        x0, y0, z0, v_prev = x, y, z, v
        a = _accel(v, k)
        vm = (v[0] + 0.5 * dt * a[0], v[1] + 0.5 * dt * a[1], v[2] + 0.5 * dt * a[2])
        am = _accel(vm, k)
        v = (v[0] + dt * am[0], v[1] + dt * am[1], v[2] + dt * am[2])
        x, y, z = x0 + dt * vm[0], y0 + dt * vm[1], z0 + dt * vm[2]

        if z < r and v[2] < 0.0:
            frac = (z0 - r) / (z0 - z)
            xb, yb = x0 + frac * (x - x0), y0 + frac * (y - y0)
            v_hit = tuple(v_prev[i] + frac * (v[i] - v_prev[i]) for i in range(3))
            v_out = reflect(v_hit)  # type: ignore[arg-type]
            bounces.append(Bounce(t_prev + frac * dt, np.array([xb, yb, 0.0]), v_hit, v_out))  # type: ignore[arg-type]
            rest = (1.0 - frac) * dt
            ab = _accel(v_out, k)
            vb = tuple(v_out[i] + 0.5 * rest * ab[i] for i in range(3))
            x, y, z = xb + rest * vb[0], yb + rest * vb[1], r + rest * vb[2]
            v = tuple(v_out[i] + rest * a_i for i, a_i in enumerate(_accel(vb, k)))  # type: ignore[assignment]

        if step % substeps == 0:
            times.append(step * dt)
            pos.append((x, y, z))
            vel.append(v)
            if y >= stop_y:
                break
    return Flight(np.array(times), np.array(pos), np.array(vel), bounces)


def ballistic_velocity(p0: np.ndarray, aim: np.ndarray, speed: float) -> Vec:
    """Launch velocity (low-angle solution) that reaches `aim` from `p0` at `speed`, no drag."""
    d = np.asarray(aim, float) - np.asarray(p0, float)
    dist = math.hypot(d[0], d[1])
    h = d[2]
    v2 = speed * speed
    disc = v2 * v2 - G * (G * dist * dist + 2.0 * h * v2)
    if disc < 0 or dist == 0:
        raise ValueError("aim point is out of reach at this launch speed")
    tan_elev = (v2 - math.sqrt(disc)) / (G * dist)
    cos_elev = 1.0 / math.sqrt(1.0 + tan_elev**2)
    horiz = speed * cos_elev
    return (horiz * d[0] / dist, horiz * d[1] / dist, speed * cos_elev * tan_elev)


class ExactPath:
    """The simulated ground-truth path, exposing the same queries as a fitted model.

    Lets the sport rules run on perfect knowledge of the trajectory to produce the true call.
    """

    def __init__(self, flight: Flight):
        self._flight = flight

    def bounce_point(self) -> np.ndarray | None:
        return self._flight.bounces[0].point if self._flight.bounces else None

    def cross_y(self, y: float) -> tuple[float, np.ndarray] | None:
        ys = self._flight.pos[:, 1]
        i = int(np.searchsorted(ys, y))
        if i == 0 or i >= len(ys):
            return None
        w = (y - ys[i - 1]) / (ys[i] - ys[i - 1])
        t = self._flight.t[i - 1] + w * (self._flight.t[i] - self._flight.t[i - 1])
        return float(t), self._flight.pos[i - 1] + w * (
            self._flight.pos[i] - self._flight.pos[i - 1]
        )
