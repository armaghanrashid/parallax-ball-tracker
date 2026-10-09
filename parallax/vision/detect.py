"""Per-frame ball candidate detection: bright, compact, round blobs."""

from __future__ import annotations

import cv2
import numpy as np

Detection = tuple[float, float, float]  # x, y (pixels, sub-pixel), approximate radius


def _centroid(xs, ys, weight: np.ndarray) -> tuple[float, float] | None:
    total = float(weight.sum())
    if total <= 0:
        return None
    return float((weight * xs).sum() / total), float((weight * ys).sum() / total)


def _ball_like(area: float, w: int, h: int, min_area, max_area, max_aspect, min_fill) -> bool:
    if not (min_area <= area <= max_area):
        return False
    return max(w, h) / min(w, h) <= max_aspect and area / (w * h) >= min_fill


def detect(
    frame: np.ndarray,
    *,
    threshold: int = 170,
    min_radius: float = 1.3,
    max_radius: float = 14.0,
    max_aspect: float = 1.7,
    min_fill: float = 0.45,
    core_fraction: float = 0.9,
) -> list[Detection]:
    """Return ball candidates as (x, y, r), largest first.

    The ball is assumed to be among the brightest objects. Pixels at or above `threshold` are
    grouped into connected blobs; blobs that are too small (hot pixels), too large (glare),
    elongated (lines, streaks) or not filled (outlines) are dropped.

    A blob can be a ball merged with a dimmer neighbour. Each blob is therefore re-thresholded at
    `core_fraction` of its own peak; when that bright core is still ball-like it replaces the
    blob. The position is the intensity-weighted centroid, accurate to a fraction of a pixel.
    """
    value = (
        frame if frame.ndim == 2 else cv2.max(cv2.max(frame[..., 0], frame[..., 1]), frame[..., 2])
    )
    mask = (value >= threshold).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area, max_area = np.pi * min_radius**2, np.pi * max_radius**2
    height, width = value.shape
    shape_args = (min_area, max_area, max_aspect, min_fill)

    found: list[tuple[float, Detection]] = []
    for i in range(1, count):
        x0, y0, w, h, area = (int(v) for v in stats[i])
        if area < min_area:
            continue
        xa, xb = max(x0 - 1, 0), min(x0 + w + 1, width)
        ya, yb = max(y0 - 1, 0), min(y0 + h + 1, height)
        window = value[ya:yb, xa:xb]

        ys, xs = np.mgrid[ya:yb, xa:xb]

        whole = None
        if _ball_like(area, w, h, *shape_args):
            whole = _centroid(xs, ys, np.clip(window.astype(np.float32) - threshold, 0.0, None))
            whole = whole and (whole, float(area))

        core = None
        peak = int(window.max())
        core_floor = max(float(threshold), core_fraction * peak)
        if core_floor > threshold:
            _, lab, st, _ = cv2.connectedComponentsWithStats(
                (window >= core_floor).astype(np.uint8), connectivity=8
            )
            py, px = np.unravel_index(int(window.argmax()), window.shape)
            k = int(lab[py, px])
            if k > 0 and _ball_like(st[k][4], st[k][2], st[k][3], *shape_args):
                keep = cv2.dilate((lab == k).astype(np.uint8), np.ones((3, 3), np.uint8))
                weight = np.clip(window.astype(np.float32) - core_floor, 0.0, None) * keep
                c = _centroid(xs, ys, weight)
                core = c and (c, float(st[k][4]))

        # A lone ball has the same centre whole or as a core. If they disagree the blob is a ball
        # merged with something dimmer, and the core is the ball.
        if whole and core and np.hypot(whole[0][0] - core[0][0], whole[0][1] - core[0][1]) <= 1.0:
            chosen = whole
        else:
            chosen = core or whole
        if chosen is None:
            continue
        (cx, cy), shape_area = chosen
        found.append((float(area), (cx, cy, float(np.sqrt(shape_area / np.pi)))))
    found.sort(key=lambda item: -item[0])
    return [d for _, d in found]
