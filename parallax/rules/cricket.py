"""Cricket LBW projection: would the ball have gone on to hit the stumps?"""

from __future__ import annotations

from dataclasses import dataclass

from parallax.rules import Decision

BALL_RADIUS = 0.0356
PITCH_LENGTH_M = 20.12


@dataclass(frozen=True)
class Stumps:
    plane_y: float = PITCH_LENGTH_M  # batter's end
    x_center: float = 0.0
    half_width: float = 0.1143  # three stumps plus gaps are 228.6 mm wide
    height: float = 0.711  # 28 in, to the top of the bails


STUMPS = Stumps()


def lbw(model, stumps: Stumps = STUMPS, ball_radius: float = BALL_RADIUS) -> Decision:
    """Project the post-impact path to the stump plane and test it against the stumps.

    The ball is HIT if any part of it overlaps the stump rectangle: its centre may be up to one
    ball radius outside the stumps. `impact` is the (x, z) of the ball centre on the plane.
    """
    crossing = model.cross_y(stumps.plane_y)
    if crossing is None:
        return Decision("MISS", None, None)
    _, xyz = crossing
    x, z = float(xyz[0]), float(xyz[2])
    slack_x = stumps.half_width + ball_radius - abs(x - stumps.x_center)
    slack_z = stumps.height + ball_radius - z
    margin_mm = 1000.0 * min(slack_x, slack_z)
    label = "HIT" if margin_mm >= -1e-9 else "MISS"
    return Decision(label, margin_mm, (x, z))
