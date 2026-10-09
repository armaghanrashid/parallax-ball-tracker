"""Hand-built trajectories shared by the fit and rules tests."""

from __future__ import annotations

import numpy as np

from parallax.vision.fit import BallisticModel, Segment

G = 9.80665


def parabola_points(t, p0, v0, a=(0.0, 0.0, -G)):
    t = np.asarray(t, dtype=float)[:, None]
    return np.asarray(p0) + np.asarray(v0) * t + 0.5 * np.asarray(a) * t**2


def single_segment_model(p0, v0, t_end=1.0, a=(0.0, 0.0, -G), ground_z=0.0335):
    seg = Segment.ballistic(0.0, t_end, p0, v0, a)
    return BallisticModel(segments=(seg,), ground_z=ground_z, bounce=None)


def bounce_model(bounce_xy, ground_z=0.0335, v_in=(1.0, 20.0, -6.0), v_out=(1.0, 17.0, 4.5)):
    """Model that touches the ground at `bounce_xy` at t = 1 s."""
    bx, by = bounce_xy
    tb = 1.0
    c = np.array([bx, by, ground_z])
    pre = Segment.ballistic(
        0.0,
        tb,
        c - np.asarray(v_in) * tb - np.array([0, 0, -G]) * 0.5 * tb**2,
        v_in,
        (0.0, 0.0, -G),
        tref=0.0,
    )
    post = Segment.ballistic(tb, tb + 1.0, c, v_out, (0.0, 0.0, -G), tref=tb)
    return BallisticModel(
        segments=(pre, post), ground_z=ground_z, bounce=(tb, np.array([bx, by, 0.0]))
    )


def bounce_samples(
    rate=240.0, ground_z=0.0335, restitution=0.7, noise=0.0, seed=0, n_pre=20, n_post=20
):
    """Noisy samples of a ball that bounces once. Returns (N, 4) [t, x, y, z] and truth."""
    rng = np.random.default_rng(seed)
    p0 = np.array([0.5, 0.0, 1.4])
    v0 = np.array([1.5, 22.0, -2.0])
    # time of impact: z0 + vz t - g t^2 / 2 = ground_z
    disc = v0[2] ** 2 + 2 * G * (p0[2] - ground_z)
    tb = (-v0[2] - np.sqrt(disc)) / -G
    pb = p0 + v0 * tb + 0.5 * np.array([0, 0, -G]) * tb**2
    vb = v0 + np.array([0, 0, -G]) * tb
    v_after = np.array([vb[0] * 0.8, vb[1] * 0.8, -vb[2] * restitution])
    dt = 1 / rate
    t_pre = tb - dt * np.arange(n_pre, 0, -1) + dt * 0.37
    t_post = tb + dt * np.arange(0, n_post) + dt * 0.37
    t_pre = t_pre[t_pre >= 0]
    pre = parabola_points(t_pre, p0, v0)
    post = parabola_points(t_post - tb, pb, v_after)
    t = np.r_[t_pre, t_post]
    pts = np.vstack([pre, post]) + rng.normal(0, noise, (len(t), 3))
    return np.c_[t, pts], {"tb": tb, "pb": np.array([pb[0], pb[1], 0.0])}
