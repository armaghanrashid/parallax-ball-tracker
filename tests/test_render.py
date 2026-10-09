import numpy as np
import pytest

from parallax.sim import simulate
from parallax.sim.render import court_geometry, render

SPORTS = ["tennis", "cricket", "football"]


@pytest.fixture(scope="module")
def tennis():
    return simulate("tennis", 3)


def test_one_frame_per_visible_sample(tennis):
    frames = render(tennis, tennis.cameras[0])
    assert len(frames) == len(tennis.visible_pos)
    assert all(f.shape == (360, 640, 3) and f.dtype == np.uint8 for f in frames)


def test_render_is_deterministic_but_noise_differs_between_frames(tennis):
    a = render(tennis, tennis.cameras[0])
    b = render(tennis, tennis.cameras[0])
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert not np.array_equal(a[0], a[1])


def test_cameras_get_independent_noise(tennis):
    a = render(tennis, tennis.cameras[0])
    b = render(tennis, tennis.cameras[1])
    assert not np.array_equal(a[0], b[0])


def test_clean_render_puts_the_ball_exactly_at_its_projection(tennis):
    cam = tennis.cameras[0]
    frames = render(tennis, cam, noise_sigma=0.0, clutter=False)
    uv = cam.project(tennis.visible_pos)
    for i in (0, len(frames) // 2, len(frames) - 1):
        v = frames[i].max(axis=2).astype(float)
        w = np.clip(v - 170.0, 0.0, None)
        ys, xs = np.mgrid[0 : v.shape[0], 0 : v.shape[1]]
        # Take the bright blob near the expected position (court lines are dimmer).
        near = (np.hypot(xs - uv[i, 0], ys - uv[i, 1]) < 8) * w
        cx, cy = (near * xs).sum() / near.sum(), (near * ys).sum() / near.sum()
        assert np.hypot(cx - uv[i, 0], cy - uv[i, 1]) < 0.15


@pytest.mark.parametrize("sport", SPORTS)
def test_ball_is_the_brightest_clean_object(sport):
    d = simulate(sport, 1)
    cam = d.cameras[1]
    frames = render(d, cam, noise_sigma=0.0, clutter=False)
    uv = cam.project(d.visible_pos)
    f = frames[len(frames) // 2].max(axis=2)
    u, v = np.round(uv[len(frames) // 2]).astype(int)
    assert f[v, u] >= 215
    f2 = f.copy()
    f2[max(v - 8, 0) : v + 9, max(u - 8, 0) : u + 9] = 0
    assert f2.max() < 170  # static structures stay below the detection threshold


def test_clutter_adds_bright_distractors_that_move(tennis):
    clean = render(tennis, tennis.cameras[0], noise_sigma=0.0, clutter=False)
    busy = render(tennis, tennis.cameras[0], noise_sigma=0.0, clutter=True)
    extra = [
        (np.abs(b.astype(int) - c.astype(int)).max(axis=2) > 40).sum()
        for b, c in zip(busy, clean, strict=True)
    ]
    assert max(extra) > 10
    assert not np.array_equal(busy[0], busy[-1])


def test_background_scrolls(tennis):
    busy = render(tennis, tennis.cameras[0], noise_sigma=0.0, clutter=True)
    top = [f[:40].astype(int) for f in busy]
    assert np.abs(top[0] - top[10]).mean() > 0.3


def test_court_geometry_has_lines_for_every_sport():
    for sport in SPORTS:
        g = court_geometry(sport)
        assert len(g.quads) >= 1
    assert len(court_geometry("cricket").poles) == 3
    assert len(court_geometry("football").poles) == 3
