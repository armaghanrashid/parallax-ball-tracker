"""Per-sport configuration shared by the pipeline, the evaluation and the Lambda handler."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from parallax.rules import Decision, cricket, football, tennis


@dataclass(frozen=True)
class Sport:
    name: str
    ball_radius: float  # m; also the ball-centre height at ground contact
    judge: Callable[..., Decision]
    detect: dict = field(default_factory=dict)  # keyword arguments for `detect`


SPORTS: dict[str, Sport] = {
    "tennis": Sport("tennis", tennis.BALL_RADIUS, tennis.call, {"max_radius": 7.0}),
    "cricket": Sport("cricket", cricket.BALL_RADIUS, cricket.lbw, {"max_radius": 8.0}),
    "football": Sport("football", football.BALL_RADIUS, football.goal_line, {"max_radius": 14.0}),
}
