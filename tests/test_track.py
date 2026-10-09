import numpy as np
import pytest

from parallax.vision.track import Tracker


def ball_path(n=80, bounce_at=40):
    """Image-space path of a ball with a bounce: vertical velocity flips at `bounce_at`."""
    t = np.arange(n, dtype=float)
    x = 100.0 + 5.0 * t
    y = np.empty(n)
    y[0] = 40.0
    vy = 1.0
    for i in range(1, n):
        vy += 0.06  # gravity-ish pull in image space
        if i == bounce_at:
            vy = -0.8 * vy
        y[i] = y[i - 1] + vy
    return np.c_[x, y]


def run(path, extra=None, noise=0.15, drop=(), seed=0, **kw):
    rng = np.random.default_rng(seed)
    tracker = Tracker(**kw)
    for i, p in enumerate(path):
        dets = [] if i in drop else [(*(p + rng.normal(0, noise, 2)), 2.5)]
        if extra:
            dets += extra(i)
        tracker.update(dets)
    return tracker


def errors(track, path):
    m = track.measured(len(path))
    ok = ~np.isnan(m[:, 0])
    return ok, np.linalg.norm(m[ok] - path[ok], axis=1)


def test_follows_a_clean_noisy_path_through_a_bounce():
    path = ball_path()
    track = run(path).best()
    ok, err = errors(track, path)
    assert ok.mean() == 1.0
    assert err.max() < 0.8


def test_rejects_a_stationary_decoy_and_a_drifting_decoy():
    path = ball_path()

    def decoys(i):
        return [(30.0, 150.0, 2.5), (200.0 + 0.4 * i, 20.0 + 0.3 * i, 2.5)]

    track = run(path, extra=decoys).best()
    ok, err = errors(track, path)
    assert ok.mean() > 0.97 and err.max() < 0.8


def test_gate_stops_the_track_hopping_to_a_lookalike_that_appears_later():
    path = ball_path()
    rng = np.random.default_rng(9)

    def shadow(i):  # 25 px away, moving in lock-step from frame 20: outside any sensible gate
        if i < 20:
            return []
        return [(*(path[i] + [25.0, -20.0] + rng.normal(0, 0.15, 2)), 2.5)]

    track = run(path, extra=shadow).best()
    ok, err = errors(track, path)
    assert ok.mean() > 0.95 and err.max() < 0.8


def test_short_lived_fast_decoy_loses_to_the_ball_track():
    path = ball_path()

    def glint(i):
        return [(300.0 - 6.0 * (i - 10), 200.0 + 3.0 * (i - 10), 2.0)] if 10 <= i < 22 else []

    track = run(path, extra=glint).best()
    ok, err = errors(track, path)
    assert ok.mean() > 0.97 and err.max() < 0.8


def test_coasts_through_missed_detections_and_reacquires():
    path = ball_path()
    track = run(path, drop=set(range(30, 34))).best()
    m = track.measured(len(path))
    assert np.isnan(m[30:34, 0]).all()
    ok, err = errors(track, path)
    assert ok[34:].all() and err.max() < 0.8


def test_reacquires_across_a_miss_that_spans_the_bounce():
    path = ball_path()
    track = run(path, drop={39, 40, 41}).best()
    ok, err = errors(track, path)
    assert ok[43:].mean() > 0.95 and err.max() < 0.8


def test_ball_that_appears_mid_clip_is_picked_up():
    path = ball_path()
    tracker = Tracker()
    rng = np.random.default_rng(3)
    for i, p in enumerate(path):
        dets = [(30.0, 150.0, 2.5)]
        if i >= 15:
            dets.append((*(p + rng.normal(0, 0.15, 2)), 2.5))
        tracker.update(dets)
    ok, err = errors(tracker.best(), path)
    assert ok[16:].all() and err.max() < 0.8


def test_no_detections_gives_no_track():
    tracker = Tracker()
    for _ in range(5):
        tracker.update([])
    assert tracker.best() is None


def test_only_static_clutter_gives_no_track():
    tracker = Tracker()
    for _ in range(20):
        tracker.update([(50.0, 50.0, 2.0)])
    assert tracker.best() is None


def test_track_state_reports_positions_for_every_frame():
    path = ball_path(30, bounce_at=99)
    track = run(path, drop={10}).best()
    assert track.filtered(len(path)).shape == (30, 2)
    assert not np.isnan(track.filtered(len(path))).any()
    assert pytest.approx(track.filtered(len(path))[10], abs=3.0) == path[10]
