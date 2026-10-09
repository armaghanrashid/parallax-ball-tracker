"""Goal-line technology: has the whole ball crossed the line, between the posts, under the bar?"""

from __future__ import annotations

from dataclasses import dataclass

from parallax.rules import Decision

BALL_RADIUS = 0.11


@dataclass(frozen=True)
class Goal:
    line_y: float = 0.0
    x_center: float = 0.0
    half_width: float = 3.66  # inner edges of the posts, 7.32 m apart
    height: float = 2.44


GOAL = Goal()


def goal_line(model, goal: Goal = GOAL, ball_radius: float = BALL_RADIUS) -> Decision:
    """GOAL when, at the moment the trailing edge of the ball crosses the line, the whole ball
    is inside the frame. `impact` is the (x, z) of the ball centre at that moment."""
    crossing = model.cross_y(goal.line_y + ball_radius)
    if crossing is None:
        return Decision("NO_GOAL", None, None)
    _, xyz = crossing
    x, z = float(xyz[0]), float(xyz[2])
    slack_x = goal.half_width - ball_radius - abs(x - goal.x_center)
    slack_z = goal.height - ball_radius - z
    margin_mm = 1000.0 * min(slack_x, slack_z)
    label = "GOAL" if margin_mm >= -1e-9 else "NO_GOAL"
    return Decision(label, margin_mm, (x, z))
