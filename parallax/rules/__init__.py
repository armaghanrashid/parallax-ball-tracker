"""Sport-specific decision rules, evaluated on a fitted trajectory.

Every rule returns a `Decision` whose `margin_mm` is positive when the call goes in favour of the
first label (IN / HIT / GOAL) and negative otherwise, so the sign always agrees with the label.
"""

from __future__ import annotations

from typing import NamedTuple


class Decision(NamedTuple):
    label: str
    margin_mm: float | None  # distance to the decision boundary; None when undefined
    impact: tuple[float, float] | None  # where the ball met the line / stump plane / goal plane
