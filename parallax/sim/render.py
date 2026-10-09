"""Synthetic camera footage: dark stadium look, painted lines, moving clutter and sensor noise."""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import cv2
import numpy as np

from parallax.camera import Camera
from parallax.rules import cricket, football, tennis

# Static structures are drawn dimmer than the ball: the detector assumes the ball is the
# brightest object in the scene (a typical floodlit-night assumption).
STRUCTURE_VALUE = 150
BALL_BGR = {
    "tennis": (60.0, 245.0, 205.0),  # optic yellow
    "cricket": (65.0, 70.0, 238.0),  # red leather
    "football": (238.0, 238.0, 235.0),  # white
}
PALETTE = {  # (top, bottom) BGR of the background gradient
    "tennis": ((30, 22, 18), (74, 48, 30)),
    "cricket": ((26, 28, 20), (44, 70, 40)),
    "football": ((24, 28, 20), (40, 66, 36)),
}


@dataclass(frozen=True)
class Geometry:
    quads: list[np.ndarray]  # (4, 3) painted shapes lying on the ground plane
    poles: list[tuple[np.ndarray, np.ndarray, float]]  # (bottom, top, diameter) in metres


def _ground_rect(x0, x1, y0, y1) -> np.ndarray:
    return np.array([[x0, y0, 0.0], [x1, y0, 0.0], [x1, y1, 0.0], [x0, y1, 0.0]])


def court_geometry(sport: str) -> Geometry:
    """Lines and poles that make up the scene, in world coordinates."""
    if sport == "tennis":
        hw, e = tennis.SINGLES_HALF_WIDTH_M, tennis.BASELINE_Y_M
        quads = [
            _ground_rect(-hw, hw, e - 0.10, e),  # far baseline (100 mm)
            _ground_rect(-hw, hw, -e, -e + 0.10),
            _ground_rect(-hw, -hw + 0.05, -e, e),  # sidelines
            _ground_rect(hw - 0.05, hw, -e, e),
            _ground_rect(-hw, hw, 6.40 - 0.025, 6.40 + 0.025),  # service line
            _ground_rect(-0.025, 0.025, 0.0, 6.40),  # centre service line
        ]
        return Geometry(quads, [])
    if sport == "cricket":
        y_s = cricket.STUMPS.plane_y
        quads = [
            _ground_rect(-1.525, 1.525, 0.0, y_s),  # pitch strip (drawn dimmer, see render)
            _ground_rect(-1.32, 1.32, y_s - 1.22 - 0.025, y_s - 1.22 + 0.025),  # popping crease
            _ground_rect(-1.32, 1.32, y_s - 0.025, y_s + 0.025),  # bowling crease
            _ground_rect(-1.32 - 0.025, -1.32 + 0.025, y_s - 1.22, y_s + 1.22),  # return creases
            _ground_rect(1.32 - 0.025, 1.32 + 0.025, y_s - 1.22, y_s + 1.22),
        ]
        s = cricket.STUMPS
        poles = [
            (np.array([x, s.plane_y, 0.0]), np.array([x, s.plane_y, s.height]), 0.038)
            for x in (-0.0762, 0.0, 0.0762)
        ]
        return Geometry(quads, poles)
    if sport == "football":
        g = football.GOAL
        hw = g.half_width + 0.06
        quads = [
            _ground_rect(-12.0, 12.0, -0.12, 0.0),  # goal line
            _ground_rect(-9.16, 9.16, -5.5 - 0.12, -5.5),  # six-yard box
            _ground_rect(-9.16 - 0.12, -9.16, -5.5, 0.0),
            _ground_rect(9.16, 9.16 + 0.12, -5.5, 0.0),
        ]
        poles = [
            (np.array([-hw, 0.06, 0.0]), np.array([-hw, 0.06, g.height + 0.12]), 0.12),
            (np.array([hw, 0.06, 0.0]), np.array([hw, 0.06, g.height + 0.12]), 0.12),
            (np.array([-hw, 0.06, g.height + 0.06]), np.array([hw, 0.06, g.height + 0.06]), 0.12),
        ]
        return Geometry(quads, poles)
    raise KeyError(sport)


def _gradient(sport: str, h: int, w: int) -> np.ndarray:
    top, bottom = (np.array(c, dtype=np.float32) for c in PALETTE[sport])
    ramp = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None, None]
    return np.broadcast_to(top + (bottom - top) * ramp, (h, w, 3)).astype(np.uint8).copy()


def draw_structures(img: np.ndarray, sport: str, cam: Camera, value: int = STRUCTURE_VALUE) -> None:
    """Draw the painted lines and poles onto `img` (BGR uint8) with anti-aliasing."""
    geo = court_geometry(sport)
    shift = 4
    scale = 1 << shift
    for i, quad in enumerate(geo.quads):
        pts = np.round(cam.project(quad) * scale).astype(np.int32)
        dim = sport == "cricket" and i == 0
        colour = (62, 84, 104) if dim else (value, value, value - 8)
        cv2.fillConvexPoly(img, pts, colour, lineType=cv2.LINE_AA, shift=shift)
    for bottom, top, diameter in geo.poles:
        a, b = cam.project(bottom), cam.project(top)
        depth = float(cam.depth(0.5 * (bottom + top)))
        width = max(1, int(round(cam.focal * diameter / depth)))
        cv2.line(
            img,
            tuple(np.round(a * scale).astype(int)),
            tuple(np.round(b * scale).astype(int)),
            (value, value, value - 8),
            width,
            cv2.LINE_AA,
            shift,
        )


def _crowd_texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Dim speckled texture for the stands, wider than the frame so it can scroll."""
    base = rng.integers(0, 60, size=(h // 3 + 1, w // 3 + 1, 1), dtype=np.uint8)
    tex = cv2.resize(base, (w, h), interpolation=cv2.INTER_NEAREST)
    tex = cv2.GaussianBlur(tex, (0, 0), 1.2)
    tint = np.array([0.7, 0.55, 0.5], dtype=np.float32)
    return (tex[..., None].astype(np.float32) * tint * 0.5).astype(np.uint8)


def _disc(img: np.ndarray, cx: float, cy: float, radius: float, bgr, antialias_shift: int = 4):
    s = 1 << antialias_shift
    cv2.circle(
        img,
        (int(round(cx * s)), int(round(cy * s))),
        int(round(radius * s)),
        bgr,
        -1,
        cv2.LINE_AA,
        antialias_shift,
    )


def _ball(img: np.ndarray, cx: float, cy: float, radius: float, bgr) -> None:
    """Analytic-coverage anti-aliased disc with mild shading, blended in a small window."""
    h, w = img.shape[:2]
    pad = int(np.ceil(radius)) + 2
    x0, x1 = max(int(cx) - pad, 0), min(int(cx) + pad + 1, w)
    y0, y1 = max(int(cy) - pad, 0), min(int(cy) + pad + 1, h)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    dist = np.hypot(xs - cx, ys - cy)
    cover = np.clip(radius + 0.5 - dist, 0.0, 1.0)[..., None]
    shade = (0.9 + 0.1 * (1.0 - np.clip(dist / radius, 0, 1) ** 2))[..., None]
    patch = img[y0:y1, x0:x1].astype(np.float32)
    colour = np.minimum(np.array(bgr, np.float32) * shade, 255.0)
    img[y0:y1, x0:x1] = np.clip(patch * (1.0 - cover) + colour * cover, 0, 255).astype(np.uint8)


class _Clutter:
    """Moving distractors: drifting lights, fast glints, a streak, players and hot pixels."""

    def __init__(self, rng: np.random.Generator, n_frames: int, cam: Camera, ball_px: float):
        self.rng, self.w, self.h = rng, cam.width, cam.height
        w, h = self.w, self.h
        n = n_frames
        self.moths = []
        for _ in range(2):
            p0 = rng.uniform([0.1 * w, 0.2 * h], [0.9 * w, 0.8 * h])
            v = rng.normal(0.0, 0.35, 2)
            r = ball_px * rng.uniform(0.8, 1.3)
            self.moths.append((p0 + v * np.arange(n)[:, None], r))
        self.glints = []
        for _ in range(2):
            start = int(rng.integers(0, max(n - 12, 1)))
            life = int(rng.integers(8, 15))
            p0 = rng.uniform([0.1 * w, 0.15 * h], [0.9 * w, 0.85 * h])
            v = rng.normal(0.0, 2.5, 2)
            self.glints.append((start, life, p0, v, ball_px * rng.uniform(0.7, 1.2)))
        self.streak = (rng.uniform([0.1 * w, 0.2 * h], [0.9 * w, 0.8 * h]), rng.normal(0, 2.0, 2))
        self.players = [
            (rng.uniform([0.1 * w, 0.45 * h], [0.9 * w, 0.8 * h]), rng.normal(0, 0.7, 2))
            for _ in range(3)
        ]

    def draw(self, img: np.ndarray, i: int) -> None:
        for pos, r in self.moths:
            _disc(img, pos[i, 0], pos[i, 1], r, (196, 204, 206))
        for start, life, p0, v, r in self.glints:
            if start <= i < start + life:
                p = p0 + v * (i - start)
                _disc(img, p[0], p[1], r, (200, 206, 200))
        p0, v = self.streak
        c = p0 + v * i
        d = np.array([6.0, 2.0])
        cv2.line(
            img,
            tuple((c - d).astype(int)),
            tuple((c + d).astype(int)),
            (190, 196, 196),
            1,
            cv2.LINE_AA,
        )
        for p0, v in self.players:
            c = p0 + v * i
            cv2.ellipse(
                img, (int(c[0]), int(c[1])), (7, 18), 0, 0, 360, (88, 92, 96), -1, cv2.LINE_AA
            )
        for _ in range(3):
            img[int(self.rng.integers(0, self.h)), int(self.rng.integers(0, self.w))] = 255


def render(
    delivery,
    camera: Camera,
    *,
    noise_sigma: float = 2.5,
    clutter: bool = True,
    seed: int | None = None,
) -> list[np.ndarray]:
    """Render the recorded frames of `delivery` as seen by `camera` (BGR uint8)."""
    sport, h, w = delivery.sport, camera.height, camera.width
    cam_key = zlib.crc32(np.round(camera.center, 6).tobytes())
    rng = np.random.default_rng([delivery.seed if seed is None else seed, cam_key])
    pos = delivery.visible_pos
    n = len(pos)
    uv = camera.project(pos)
    radius_px = camera.focal * delivery.ball_radius / camera.depth(pos)

    base = _gradient(sport, h, w)
    draw_structures(base, sport, camera)
    stands_h = int(0.28 * h)
    crowd = _crowd_texture(rng, stands_h, w + 2 * n + 8) if clutter else None
    scroll = rng.uniform(0.6, 1.4)
    things = _Clutter(rng, n, camera, float(radius_px.mean())) if clutter else None
    if noise_sigma > 0:
        canvas = rng.normal(0.0, noise_sigma, (h + 16, w + 16, 3)).round().astype(np.int16)
        canvas_pos = np.clip(canvas, 0, 255).astype(np.uint8)
        canvas_neg = np.clip(-canvas, 0, 255).astype(np.uint8)

    frames = []
    for i in range(n):
        img = base.copy()
        if clutter:
            off = int(i * scroll)
            img[:stands_h] = cv2.add(img[:stands_h], crowd[:, off : off + w])
            things.draw(img, i)
        _ball(img, float(uv[i, 0]), float(uv[i, 1]), float(radius_px[i]), BALL_BGR[sport])
        if noise_sigma > 0:
            dy, dx = (int(v) for v in rng.integers(0, 16, 2))
            img = cv2.add(img, np.ascontiguousarray(canvas_pos[dy : dy + h, dx : dx + w]))
            img = cv2.subtract(img, np.ascontiguousarray(canvas_neg[dy : dy + h, dx : dx + w]))
        frames.append(img)
    return frames
