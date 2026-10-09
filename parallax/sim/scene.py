"""Scenario generation: ball launch, court geometry, camera rig and the true call."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from parallax.camera import Camera
from parallax.rules import Decision, cricket, football, tennis
from parallax.sim.physics import (
    SPECS,
    BallSpec,
    ExactPath,
    Flight,
    ballistic_velocity,
    integrate,
)

IMAGE_SIZE = (640, 360)
SHOOT_SUBSTEPS = 4
FLIGHT_SUBSTEPS = 10


class _Rejected(Exception):
    """The sampled scenario is unusable (e.g. unwanted bounce); sample another."""


@dataclass(frozen=True, eq=False)
class Delivery:
    """One simulated ball flight, with its camera rig and the ground-truth call."""

    sport: str
    seed: int
    flight: Flight  # full trajectory, 240 Hz
    window: slice  # frames the cameras record
    cameras: tuple[Camera, Camera]
    truth: Decision  # the rules applied to the exact path
    ball_radius: float

    @property
    def visible_t(self) -> np.ndarray:
        return self.flight.t[self.window]

    @property
    def visible_pos(self) -> np.ndarray:
        return self.flight.pos[self.window]


def _shoot(
    spec: BallSpec,
    p0: np.ndarray,
    speed: float,
    target: np.ndarray,
    locate: Callable[[Flight], np.ndarray],
    stop_y: float,
    bounce_fn=None,
    iterations: int = 8,
) -> tuple[float, float, float]:
    """Find a launch velocity whose *drag-affected* flight passes through `target`.

    Starts from the drag-free ballistic solution and repeatedly shifts the aim point by the miss.
    """
    aim = target.copy()
    v0 = ballistic_velocity(p0, aim, speed)
    for _ in range(iterations):
        v0 = ballistic_velocity(p0, aim, speed)
        flight = integrate(
            spec, tuple(p0), v0, stop_y=stop_y, substeps=SHOOT_SUBSTEPS, bounce_fn=bounce_fn
        )
        miss = target - locate(flight)
        if np.abs(miss).max() < 5e-4:
            break
        aim = aim + miss
    return v0


def _window(flight: Flight, y_from: float, y_to: float) -> slice:
    ys = flight.pos[:, 1]
    start = int(np.searchsorted(ys, y_from))
    stop = int(np.searchsorted(ys, y_to, side="right"))
    return slice(start, stop)


def _recorded(cams, flight: Flight, window: slice, anchor: int, margin: float = 8.0) -> slice:
    """Trim `window` to the contiguous run, around frame `anchor`, seen by both cameras."""
    idx = np.arange(window.start, window.stop)
    seen = cams[0].in_frame(flight.pos[idx], margin) & cams[1].in_frame(flight.pos[idx], margin)
    k = int(np.clip(anchor - window.start, 0, len(idx) - 1))
    if not seen[k]:
        raise _Rejected("anchor frame is outside the camera views")
    lo = hi = k
    while lo > 0 and seen[lo - 1]:
        lo -= 1
    while hi < len(idx) - 1 and seen[hi + 1]:
        hi += 1
    return slice(window.start + lo, window.start + hi + 1)


def _jitter(rng: np.random.Generator, pos, scale: float = 0.3) -> np.ndarray:
    return np.asarray(pos, float) + rng.normal(0.0, scale, 3)


def _tennis(rng: np.random.Generator):
    spec = SPECS["tennis"]
    x_b = rng.uniform(-3.2, 3.2)
    y_b = tennis.BASELINE_Y_M + rng.uniform(-0.11, 0.11)
    p0 = np.array([rng.uniform(-2.5, 2.5), -10.5, rng.uniform(0.7, 1.3)])
    speed = rng.uniform(28.0, 38.0)
    target = np.array([x_b, y_b, spec.radius])
    stop_y = y_b + 3.6

    def locate(fl):
        b = fl.bounces[0].point if fl.bounces else np.array([0.0, 0.0, 0.0])
        return np.array([b[0], b[1], spec.radius])

    v0 = _shoot(spec, p0, speed, target, locate, stop_y)
    flight = integrate(spec, tuple(p0), v0, stop_y=stop_y, substeps=FLIGHT_SUBSTEPS)
    aim = (x_b, tennis.BASELINE_Y_M, 0.6)
    cams = (
        Camera.look_at(_jitter(rng, (-8.0, 22.0, 4.0)), aim, 1000.0, IMAGE_SIZE),
        Camera.look_at(_jitter(rng, (8.0, 21.0, 4.8)), aim, 1000.0, IMAGE_SIZE),
    )
    window = _window(flight, y_b - 3.0, y_b + 1.6)
    window = _recorded(cams, flight, window, int(np.searchsorted(flight.t, flight.bounces[0].t)))
    return spec, flight, window, cams, (lambda m: tennis.call(m))


def _cricket(rng: np.random.Generator):
    spec = SPECS["cricket"]
    stump_y = cricket.STUMPS.plane_y
    x_b = rng.uniform(-0.3, 0.3)
    y_b = stump_y - rng.uniform(4.5, 7.5)
    x_stump = rng.uniform(-0.3, 0.3)  # where the post-bounce path is steered to
    p0 = np.array([rng.uniform(-0.4, 0.4), 1.0, rng.uniform(2.0, 2.4)])
    speed = rng.uniform(30.0, 40.0)
    pitch_character = rng.uniform(0.42, 0.6)  # restitution varies with the surface

    def bounce(v):
        vy = spec.retention * v[1]
        vx = (x_stump - x_b) / ((stump_y - y_b) / vy)
        return (vx, vy, -pitch_character * v[2])

    target = np.array([x_b, y_b, spec.radius])

    def locate(fl):
        b = fl.bounces[0].point if fl.bounces else np.zeros(3)
        return np.array([b[0], b[1], spec.radius])

    v0 = _shoot(spec, p0, speed, target, locate, y_b + 0.5, bounce_fn=bounce)
    flight = integrate(
        spec, tuple(p0), v0, stop_y=stump_y + 0.1, substeps=FLIGHT_SUBSTEPS, bounce_fn=bounce
    )
    y_pad = stump_y - rng.uniform(0.9, 1.6)  # the pad is struck here; footage ends
    aim = (0.0, 15.5, 0.9)
    cams = (
        Camera.look_at(_jitter(rng, (-11.0, 15.5, 2.5)), aim, 800.0, IMAGE_SIZE),
        Camera.look_at(_jitter(rng, (0.5, 28.0, 3.5)), aim, 1000.0, IMAGE_SIZE),
    )
    window = _window(flight, 10.5, y_pad)
    window = _recorded(cams, flight, window, int(np.searchsorted(flight.t, flight.bounces[0].t)))
    return spec, flight, window, cams, (lambda m: cricket.lbw(m))


def _football(rng: np.random.Generator):
    spec = SPECS["football"]
    r = spec.radius
    mode = rng.choice(["post", "bar", "open"], p=[0.35, 0.3, 0.35])
    if mode == "post":
        x_t = rng.choice([-1.0, 1.0]) * (football.GOAL.half_width - r + rng.uniform(-0.12, 0.12))
        z_t = rng.uniform(0.35, 2.0)
    elif mode == "bar":
        x_t = rng.uniform(-3.0, 3.0)
        z_t = football.GOAL.height - r + rng.uniform(-0.12, 0.12)
    else:
        x_t = rng.uniform(-4.5, 4.5)
        z_t = rng.uniform(0.35, 3.0)
    p0 = np.array([rng.uniform(-7.0, 7.0), -rng.uniform(14.0, 20.0), r])
    speed = rng.uniform(20.0, 30.0)
    target = np.array([x_t, r, z_t])

    def locate(fl):
        ys = fl.pos[:, 1]
        i = min(int(np.searchsorted(ys, r)), len(ys) - 1)
        return np.array([fl.pos[i, 0], r, fl.pos[i, 2]])

    v0 = _shoot(spec, p0, speed, target, locate, r + 0.3)
    flight = integrate(spec, tuple(p0), v0, stop_y=1.0, substeps=FLIGHT_SUBSTEPS)
    if flight.bounces:
        raise _Rejected("shot bounces before the line")
    aim = (0.0, -2.0, 1.2)
    cams = (
        Camera.look_at(_jitter(rng, (-10.0, 5.5, 3.2)), aim, 800.0, IMAGE_SIZE),
        Camera.look_at(_jitter(rng, (10.0, 5.5, 3.2)), aim, 800.0, IMAGE_SIZE),
    )
    window = _window(flight, -5.5, 0.9)
    window = _recorded(cams, flight, window, int(np.searchsorted(flight.pos[:, 1], 0.0)))
    return spec, flight, window, cams, (lambda m: football.goal_line(m))


_BUILDERS = {"tennis": _tennis, "cricket": _cricket, "football": _football}


def simulate(sport: str, seed: int) -> Delivery:
    """Simulate one delivery for `sport`. Deterministic in (sport, seed)."""
    build = _BUILDERS[sport]
    for attempt in range(50):
        rng = np.random.default_rng([seed, sorted(_BUILDERS).index(sport), attempt])
        try:
            spec, flight, window, cams, judge = build(rng)
        except _Rejected:
            continue
        return Delivery(
            sport=sport,
            seed=seed,
            flight=flight,
            window=window,
            cameras=cams,
            truth=judge(ExactPath(flight)),
            ball_radius=spec.radius,
        )
    raise RuntimeError(f"no usable {sport} scenario found for seed {seed}")
