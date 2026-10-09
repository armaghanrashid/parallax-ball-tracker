"""Tennis line calls: where did the ball land relative to a painted line?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from parallax.rules import Decision

BALL_RADIUS = 0.0335
# A ball is IN if any part of its footprint touches the line. The contact patch of a hard-hit
# ball is a few centimetres across; 20 mm is the radius used here.
FOOTPRINT_M = 0.020
BASELINE_Y_M = 11.885  # outer edge of the far baseline, measured from the net
SINGLES_HALF_WIDTH_M = 4.115
HALF_LINE_WIDTH_M = 0.05  # baselines are 100 mm wide


class NoBounce(ValueError):
    """The fitted trajectory has no bounce, so there is nothing to call."""


@dataclass(frozen=True)
class Line:
    """A painted line segment on the ground plane. Coordinates are (x, y) in metres."""

    p0: tuple[float, float]
    p1: tuple[float, float]
    half_width: float
    inward: tuple[float, float]  # unit normal pointing into the court


BASELINE = Line(
    p0=(-SINGLES_HALF_WIDTH_M, BASELINE_Y_M - HALF_LINE_WIDTH_M),
    p1=(SINGLES_HALF_WIDTH_M, BASELINE_Y_M - HALF_LINE_WIDTH_M),
    half_width=HALF_LINE_WIDTH_M,
    inward=(0.0, -1.0),
)


def call(model, line: Line = BASELINE, footprint: float = FOOTPRINT_M) -> Decision:
    """IN if any part of the ball's footprint touches the line or the court side of it."""
    bounce = model.bounce_point()
    if bounce is None:
        raise NoBounce("trajectory has no bounce")
    p0, p1 = np.asarray(line.p0, float), np.asarray(line.p1, float)
    centre = 0.5 * (p0 + p1)
    length = float(np.linalg.norm(p1 - p0))
    direction = (p1 - p0) / length
    outward = -np.asarray(line.inward, float)
    rel = np.asarray(bounce[:2], float) - centre

    across = line.half_width + footprint - float(rel @ outward)  # > 0: footprint reaches the line
    along = 0.5 * length + footprint - abs(float(rel @ direction))  # > 0: not past the line's end
    margin_mm = 1000.0 * min(across, along)
    label = "IN" if margin_mm >= -1e-9 else "OUT"
    return Decision(label, margin_mm, (float(bounce[0]), float(bounce[1])))
