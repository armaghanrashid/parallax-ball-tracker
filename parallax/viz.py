"""Third-umpire style visualisation: both camera views, track overlays, 3D inset and the call."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from parallax.camera import Camera
from parallax.pipeline import Analysis
from parallax.rules import cricket, football, tennis
from parallax.sim.render import court_geometry
from parallax.sports import SPORTS
from parallax.vision.detect import detect

W, H = 900, 560
PANEL_W, PANEL_H = 440, 248
INSET = (6, 316, 440, 238)  # x, y, w, h of the 3D view
BG = (13, 15, 19)
SURFACE = (22, 26, 32)
LINE = (52, 62, 74)
TEXT = (232, 236, 240)
MUTED = (132, 144, 158)
TEAL = (64, 201, 176)
AMBER = (245, 190, 70)
GOOD = (86, 204, 126)
BAD = (240, 98, 98)
POSITIVE = {"IN", "HIT", "GOAL"}
CAPTION = {
    "IN": "IN",
    "OUT": "OUT",
    "HIT": "HITTING",
    "MISS": "MISSING",
    "GOAL": "GOAL",
    "NO_GOAL": "NO GOAL",
    "NO_DECISION": "NO CALL",
}
INSET_VIEW = {  # (offset from the path centre to the virtual camera, focal px)
    "tennis": ((-4.0, -6.5, 3.4), 400.0),
    "cricket": ((-4.5, -6.0, 2.6), 380.0),
    "football": ((6.5, -7.5, 3.0), 320.0),
}
GROUND = {  # (x_min, x_max, y_min, y_max) of the pitch drawn in the 3D view
    "tennis": (-5.5, 5.5, -2.0, 14.5),
    "cricket": (-2.5, 2.5, 8.0, 22.0),
    "football": (-12.0, 12.0, -8.0, 3.0),
}
QUANT = 10


def _font(name: str, size: int) -> ImageFont.FreeTypeFont:
    path = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name
    return ImageFont.truetype(str(path), size)


class _Fonts:
    def __init__(self):
        self.title = _font("DejaVuSans-Bold.ttf", 19)
        self.small = _font("DejaVuSans.ttf", 12)
        self.tiny = _font("DejaVuSans.ttf", 11)
        self.mono = _font("DejaVuSansMono.ttf", 12)
        self.call = _font("DejaVuSans-Bold.ttf", 62)
        self.sub = _font("DejaVuSans.ttf", 16)
        self.caption = _font("DejaVuSans-Bold.ttf", 12)


SHIFT = 4
ONE = 1 << SHIFT


def _pt(p) -> tuple[int, int]:
    return int(round(float(p[0]) * ONE)), int(round(float(p[1]) * ONE))


def _ring(img, centre, radius, colour, thickness=1):
    cv2.circle(img, _pt(centre), int(round(radius * ONE)), colour, thickness, cv2.LINE_AA, SHIFT)


def _dot(img, centre, radius, colour):
    cv2.circle(img, _pt(centre), int(round(radius * ONE)), colour, -1, cv2.LINE_AA, SHIFT)


def _poly(img, pts, colour, thickness=1, closed=False):
    arr = np.array([_pt(p) for p in pts], np.int32)
    cv2.polylines(img, [arr], closed, colour, thickness, cv2.LINE_AA, SHIFT)


def reveal_frame(sport: str, analysis: Analysis, n_frames: int) -> int:
    """Frame at which the decision becomes available."""
    model = analysis.model
    if model is None:
        return n_frames - 1
    if sport == "tennis" and model.bounce is not None:
        return min(int(np.ceil(model.bounce[0] * 240.0)), n_frames - 1)
    if sport == "football":
        cross = model.cross_y(football.GOAL.line_y + football.BALL_RADIUS)
        if cross is not None:
            return min(int(np.ceil(cross[0] * 240.0)), n_frames - 1)
    return n_frames - 1


def _fitted_path(sport: str, analysis: Analysis) -> np.ndarray:
    model = analysis.model
    t0, t1 = analysis.points[0, 0], analysis.points[-1, 0]
    if sport == "cricket":
        cross = model.cross_y(cricket.STUMPS.plane_y)
        if cross is not None:
            t1 = max(t1, cross[0])
    return np.array([model.position(t) for t in np.arange(t0, t1 + 1 / 480, 1 / 480)])


def _headline(sport: str, analysis: Analysis) -> tuple[str, str]:
    d = analysis.decision
    if d.margin_mm is None:
        return CAPTION.get(d.label, d.label), "no decision"
    m = abs(d.margin_mm)
    sub = {
        "tennis": f"{'touching' if d.label == 'IN' else 'outside'} the line by {m:.0f} mm",
        "cricket": f"{'inside' if d.label == 'HIT' else 'outside'} the stumps by {m:.0f} mm",
        "football": f"{'inside' if d.label == 'GOAL' else 'outside'} the frame by {m:.0f} mm",
    }[sport]
    return CAPTION[d.label], sub


class Scene:
    """Pre-computes everything that does not change from frame to frame."""

    def __init__(self, delivery, frames, analysis: Analysis):
        self.delivery, self.analysis = delivery, analysis
        self.sport = delivery.sport
        self.cfg = SPORTS[self.sport]
        self.n = len(frames[0])
        self.rgb = [
            [cv2.GaussianBlur(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (0, 0), 1.3) for f in cam]
            for cam in frames
        ]
        self.crops = [self._crop(delivery.cameras[c], delivery.visible_pos) for c in range(2)]
        self.detections = [[detect(f, **self.cfg.detect) for f in cam] for cam in frames]
        offset, focal = INSET_VIEW[self.sport]
        centre = delivery.visible_pos.mean(axis=0) * np.array([1.0, 1.0, 0.3])
        self.vcam = Camera.look_at(centre + np.array(offset), centre, focal, (INSET[2], INSET[3]))
        self.fonts = _Fonts()
        self.reveal = reveal_frame(self.sport, analysis, self.n)
        self.path = _fitted_path(self.sport, analysis) if analysis.model else None
        self.geo = court_geometry(self.sport)
        truth = delivery.visible_pos[analysis.frame_index]
        err = np.linalg.norm(analysis.points[:, 1:] - truth, axis=1)
        self.rmse_mm = float(np.sqrt((err**2).mean())) * 1000.0 if len(err) else float("nan")
        self.label, self.sub = _headline(self.sport, analysis)
        self.colour = GOOD if analysis.decision.label in POSITIVE else BAD
        self.step = 2 if self.n > 90 else 1  # long clips are decimated to keep the GIF small
        self.base = self._base()

    @staticmethod
    def _crop(cam: Camera, pos: np.ndarray) -> tuple[float, float, float]:
        """Window (x0, y0, width) of the camera image that frames the ball's whole path."""
        uv = cam.project(pos)
        lo, hi = uv.min(axis=0) - 70.0, uv.max(axis=0) + 70.0
        width = max(hi[0] - lo[0], (hi[1] - lo[1]) * PANEL_W / PANEL_H, 330.0)
        width = min(width, float(cam.width))
        height = width * PANEL_H / PANEL_W
        cx, cy = 0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1])
        x0 = float(np.clip(cx - width / 2, 0, cam.width - width))
        y0 = float(np.clip(cy - height / 2, 0, cam.height - height))
        return x0, y0, width

    # ---------- static layer ----------
    def _base(self) -> np.ndarray:
        img = np.full((H, W, 3), BG, np.uint8)
        for x in (6, 454):
            cv2.rectangle(img, (x - 1, 49), (x + PANEL_W, 49 + PANEL_H), LINE, 1)
        x, y, w, h = INSET
        cv2.rectangle(img, (x - 1, y - 1), (x + w, y + h), LINE, 1)
        cv2.rectangle(img, (x, y), (x + w - 1, y + h - 1), SURFACE, -1)
        cv2.rectangle(img, (454 - 1, 316 - 1), (454 + 440, 316 + 238), LINE, 1)
        return img

    # ---------- per frame ----------
    def frame(self, i: int) -> np.ndarray:
        img = self.base.copy()
        for c, x0 in enumerate((6, 454)):
            self._panel(img, c, i, x0, 49)
        self._inset(img, i)
        pil = Image.fromarray(img)
        self._text(ImageDraw.Draw(pil), i)
        return np.asarray(pil)

    def _panel(self, img, cam_index: int, i: int, x0: int, y0: int) -> None:
        cx, cy, cw = self.crops[cam_index]
        ch = cw * PANEL_H / PANEL_W
        scale = PANEL_W / cw
        src = self.rgb[cam_index][i]
        matrix = np.array([[scale, 0.0, -cx * scale], [0.0, scale, -cy * scale]])
        local = cv2.warpAffine(src, matrix, (PANEL_W, PANEL_H), flags=cv2.INTER_CUBIC)
        local = (local // QUANT) * QUANT  # flatten sensor noise so the GIF compresses
        origin = np.array([cx, cy])
        del ch

        track = self.analysis.tracks[cam_index]
        tracked = track[i] if i < len(track) else np.array([np.nan, np.nan])

        def to_panel(p) -> np.ndarray:
            return (np.asarray(p, float) - origin) * scale

        for x, y, r in self.detections[cam_index][i]:
            if not np.isnan(tracked[0]) and np.hypot(x - tracked[0], y - tracked[1]) < 2.0:
                continue
            _ring(local, to_panel((x, y)), max(r * scale, 1.5) + 3.5, MUTED, 1)

        start = max(0, i - 30)
        pts = [(j, track[j]) for j in range(start, i + 1) if not np.isnan(track[j][0])]
        for (_, a), (jb, b) in zip(pts[:-1], pts[1:], strict=True):
            age = (i - jb) / 30.0
            colour = tuple(int(v * (1.0 - 0.75 * age)) for v in TEAL)
            cv2.line(local, _pt(to_panel(a)), _pt(to_panel(b)), colour, 2, cv2.LINE_AA, SHIFT)
        if not np.isnan(tracked[0]):
            cam = self.delivery.cameras[cam_index]
            depth = float(cam.depth(self.delivery.visible_pos[i]))
            radius = max(cam.focal * self.delivery.ball_radius / depth * scale, 1.5) + 5.0
            _ring(local, to_panel(tracked), radius, TEAL, 2)
        img[y0 : y0 + PANEL_H, x0 : x0 + PANEL_W] = local

    def _inset(self, img: np.ndarray, i: int) -> None:
        x0, y0, w, h = INSET
        view = np.full((h, w, 3), SURFACE, np.uint8)
        cam = self.vcam
        x_lo, x_hi, y_lo, y_hi = GROUND[self.sport]
        ground = np.array(
            [[x_lo, y_lo, 0], [x_hi, y_lo, 0], [x_hi, y_hi, 0], [x_lo, y_hi, 0]], float
        )
        cv2.fillConvexPoly(
            view,
            np.array([_pt(p) for p in cam.project(ground)], np.int32),
            (26, 34, 44),
            cv2.LINE_AA,
            SHIFT,
        )
        for quad in self.geo.quads:
            _poly(view, cam.project(quad), LINE, 1, closed=True)
        for bottom, top, diameter in self.geo.poles:
            width = max(1, int(round(cam.focal * diameter / float(cam.depth(bottom)))))
            cv2.line(
                view,
                _pt(cam.project(bottom)),
                _pt(cam.project(top)),
                (150, 160, 172),
                width,
                cv2.LINE_AA,
                SHIFT,
            )

        reveal = i >= self.reveal
        a = self.analysis
        if self.path is not None and len(self.path):
            proj = cam.project(self.path)
            t_now = i / 240.0
            n_now = int(np.clip((t_now - a.points[0, 0]) * 480.0, 0, len(proj) - 1))
            _poly(view, proj[: n_now + 1], tuple(int(v * 0.8) for v in AMBER), 1)
            if reveal and self.sport == "cricket" and n_now < len(proj) - 1:
                self._dashed(view, proj[n_now:], AMBER)
        for p in a.points[a.points[:, 0] <= i / 240.0 + 1e-9]:
            _dot(view, cam.project(p[1:]), 2.0, AMBER)
        if i < len(self.delivery.visible_pos):
            c = cam.project(self.delivery.visible_pos[i])
            radius = max(
                cam.focal
                * self.delivery.ball_radius
                / float(cam.depth(self.delivery.visible_pos[i])),
                2.5,
            )
            _ring(view, c, radius + 2.0, TEXT, 1)
        if reveal and a.model is not None:
            self._marker(view, cam)
        img[y0 : y0 + h, x0 : x0 + w] = view

    @staticmethod
    def _dashed(view, pts, colour, on=4, off=4):
        for k, (p, q) in enumerate(zip(pts[:-1], pts[1:], strict=True)):
            if (k // on) % 2 == 0:
                cv2.line(view, _pt(p), _pt(q), colour, 1, cv2.LINE_AA, SHIFT)

    def _marker(self, view, cam: Camera) -> None:
        model, sport = self.analysis.model, self.sport
        if sport == "tennis" and model.bounce_point() is not None:
            b = model.bounce_point()
            r = 0.22
            ring = np.array(
                [[b[0] + r * np.cos(a), b[1] + r * np.sin(a), 0.0] for a in np.linspace(0, 6.3, 40)]
            )
            _poly(view, cam.project(ring), self.colour, 2)
            line = tennis.BASELINE
            edge = np.array(
                [[line.p0[0], tennis.BASELINE_Y_M, 0], [line.p1[0], tennis.BASELINE_Y_M, 0]]
            )
            _poly(view, cam.project(edge), TEXT, 2)
        elif sport == "cricket":
            s = cricket.STUMPS
            frame = np.array(
                [
                    [s.x_center - s.half_width, s.plane_y, 0],
                    [s.x_center + s.half_width, s.plane_y, 0],
                    [s.x_center + s.half_width, s.plane_y, s.height],
                    [s.x_center - s.half_width, s.plane_y, s.height],
                ]
            )
            _poly(view, cam.project(frame), TEXT, 1, closed=True)
            if self.analysis.decision.impact:
                x, z = self.analysis.decision.impact
                _dot(view, cam.project(np.array([x, s.plane_y, z])), 4.0, self.colour)
                _ring(view, cam.project(np.array([x, s.plane_y, z])), 8.0, self.colour, 2)
        elif sport == "football":
            g = football.GOAL
            frame = np.array(
                [
                    [-g.half_width, g.line_y, 0],
                    [-g.half_width, g.line_y, g.height],
                    [g.half_width, g.line_y, g.height],
                    [g.half_width, g.line_y, 0],
                ]
            )
            _poly(view, cam.project(frame), TEXT, 1)
            if self.analysis.decision.impact:
                x, z = self.analysis.decision.impact
                p = cam.project(np.array([x, g.line_y + football.BALL_RADIUS, z]))
                _dot(view, p, 3.0, self.colour)
                _ring(view, p, 9.0, self.colour, 2)

    # ---------- text layer ----------
    def _text(self, d: ImageDraw.ImageDraw, i: int) -> None:
        f = self.fonts
        d.text((10, 12), "PARALLAX", font=f.title, fill=TEXT)
        d.text((128, 17), "two-camera ball tracking", font=f.small, fill=MUTED)
        slow = 240.0 / 25.0 / self.step
        right = f"{self.sport.upper()}   240 Hz capture, {slow:.1f}x slow motion"
        d.text((W - 10, 17), right, font=f.small, fill=MUTED, anchor="ra")
        d.text((10, 300), "CAMERA A", font=f.caption, fill=MUTED)
        d.text((458, 300), "CAMERA B", font=f.caption, fill=MUTED)
        x0, y0, w, h = INSET
        d.text((x0 + 8, y0 + 7), "RECONSTRUCTED 3D PATH", font=f.caption, fill=MUTED)

        px, py = 454, 316
        d.text((px + 12, py + 9), "DECISION", font=f.caption, fill=MUTED)
        if i >= self.reveal and self.analysis.model is not None:
            d.text((px + 12, py + 28), self.label, font=f.call, fill=self.colour)
            d.text((px + 14, py + 104), self.sub, font=f.sub, fill=TEXT)
        else:
            d.text((px + 12, py + 38), "TRACKING", font=f.call, fill=(70, 80, 92))
            d.text((px + 14, py + 104), "waiting for the decisive moment", font=f.sub, fill=MUTED)

        n_pts = int((self.analysis.points[:, 0] <= i / 240.0 + 1e-9).sum())
        rows = [
            f"frame {i + 1:03d}/{self.n:03d}   t = {i / 240 * 1000:5.0f} ms",
            f"triangulated points {n_pts:3d}   RMSE {self.rmse_mm:4.1f} mm",
            f"analysis latency {self.analysis.latency_ms:4.0f} ms per clip",
        ]
        for k, row in enumerate(rows):
            d.text((px + 14, py + 138 + 17 * k), row, font=f.mono, fill=MUTED)

        ly = py + 238 - 26
        lx = px + 14
        items = [
            ("ring", TEAL, "tracked ball"),
            ("ring", MUTED, "rejected"),
            ("dot", AMBER, "3D point"),
            ("line", tuple(int(v * 0.8) for v in AMBER), "fitted path"),
        ]
        for kind, colour, text in items:
            cy = ly + 8
            if kind == "ring":
                d.ellipse((lx, cy - 5, lx + 10, cy + 5), outline=colour, width=2)
            elif kind == "dot":
                d.ellipse((lx + 2, cy - 3, lx + 8, cy + 3), fill=colour)
            else:
                d.line((lx, cy, lx + 12, cy), fill=colour, width=2)
            d.text((lx + 17, ly), text, font=f.tiny, fill=MUTED)
            lx += 17 + int(d.textlength(text, font=f.tiny)) + 14


def build_frames(delivery, frames, analysis: Analysis, hold: int = 30) -> list[np.ndarray]:
    scene = Scene(delivery, frames, analysis)
    indices = sorted({*range(0, scene.n, scene.step), scene.n - 1})
    out = [scene.frame(i) for i in indices]
    out.extend([out[-1]] * hold)
    return out


def write_gif(frames: list[np.ndarray], path: str | Path, fps: int = 25) -> int:
    """Encode with ffmpeg palettegen/paletteuse. Returns the file size in bytes."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to encode GIFs")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for k, frame in enumerate(frames):
            Image.fromarray(frame).save(tmp_dir / f"f{k:04d}.png")
        pattern = str(tmp_dir / "f%04d.png")
        palette = str(tmp_dir / "palette.png")
        run = ["ffmpeg", "-v", "error", "-y", "-framerate", str(fps), "-i", pattern]
        subprocess.run(
            [*run, "-vf", "palettegen=max_colors=64:stats_mode=diff", palette], check=True
        )
        subprocess.run(
            [
                *run,
                "-i",
                palette,
                "-lavfi",
                "paletteuse=dither=none:diff_mode=rectangle",
                "-loop",
                "0",
                str(path),
            ],
            check=True,
        )
    return path.stat().st_size
