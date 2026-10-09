import cv2
import numpy as np
import pytest

from parallax.sim import simulate
from parallax.sim.render import render
from parallax.vision.detect import detect


def blank(h=120, w=160, level=30):
    return np.full((h, w, 3), level, np.uint8)


def draw_disc(img, cx, cy, r, value=235):
    s = 1 << 4
    cv2.circle(img, (round(cx * s), round(cy * s)), round(r * s), (value,) * 3, -1, cv2.LINE_AA, 4)


def test_finds_a_disc_with_subpixel_accuracy():
    errs = []
    for fx in np.linspace(0.0, 0.9, 4):
        for fy in np.linspace(0.0, 0.9, 4):
            img = blank()
            draw_disc(img, 60 + fx, 50 + fy, 3.0)
            ((x, y, r),) = detect(img)
            errs.append(np.hypot(x - (60 + fx), y - (50 + fy)))
            assert abs(r - 3.0) < 0.8
    assert max(errs) < 0.4


def test_subpixel_accuracy_survives_sensor_noise():
    rng = np.random.default_rng(0)
    errs = []
    for _ in range(100):
        cx, cy = rng.uniform(40, 120), rng.uniform(30, 90)
        img = blank()
        draw_disc(img, cx, cy, 2.6)
        noisy = np.clip(img + rng.normal(0, 3.0, img.shape), 0, 255).astype(np.uint8)
        found = detect(noisy)
        assert len(found) == 1
        errs.append(np.hypot(found[0][0] - cx, found[0][1] - cy))
    assert np.sqrt(np.mean(np.square(errs))) < 0.25


def test_rejects_hot_pixels_lines_and_glare():
    img = blank()
    img[20, 20] = 255  # hot pixel
    cv2.line(img, (10, 90), (70, 95), (230, 230, 230), 1, cv2.LINE_AA)  # streak
    draw_disc(img, 110, 40, 22.0)  # glare far larger than a ball
    cv2.rectangle(img, (120, 80), (150, 84), (230, 230, 230), -1)  # elongated bar
    assert detect(img) == []


def test_returns_every_ball_like_blob():
    img = blank()
    draw_disc(img, 40, 40, 3.0)
    draw_disc(img, 100, 70, 2.5)
    found = detect(img)
    assert len(found) == 2
    assert sorted(round(d[0]) for d in found) == [40, 100]


def test_dimmer_court_line_under_the_ball_is_ignored_and_ball_stays_accurate():
    img = blank()
    cv2.line(img, (0, 50), (159, 50), (150, 150, 142), 4, cv2.LINE_AA)
    draw_disc(img, 80.4, 50.3, 3.0)
    found = detect(img)
    assert len(found) == 1
    assert np.hypot(found[0][0] - 80.4, found[0][1] - 50.3) < 0.35


def test_accepts_grayscale_frames():
    img = blank()[..., 0].copy()
    cv2.circle(img, (50, 50), 3, 235, -1)
    assert len(detect(img)) == 1


@pytest.mark.parametrize("sport", ["tennis", "cricket", "football"])
def test_finds_the_ball_in_rendered_footage_with_clutter(sport):
    d = simulate(sport, 2)
    cam = d.cameras[0]
    frames = render(d, cam)
    uv = cam.project(d.visible_pos)
    hit, errs = 0, []
    for frame, truth in zip(frames, uv, strict=True):
        dets = detect(frame)
        if dets:
            best = min(dets, key=lambda q: np.hypot(q[0] - truth[0], q[1] - truth[1]))
            e = np.hypot(best[0] - truth[0], best[1] - truth[1])
            if e < 1.5:
                hit += 1
                errs.append(e)
    assert hit / len(frames) > 0.9
    assert np.sqrt(np.mean(np.square(errs))) < 0.35
