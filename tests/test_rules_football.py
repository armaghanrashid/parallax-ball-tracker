import numpy as np
import pytest

from parallax.rules import football
from parallax.rules.football import BALL_RADIUS, GOAL
from parallax.vision.fit import BallisticModel, Segment

G = 9.80665


def shot(x_cross, z_cross, vy=22.0, vx=0.0):
    """Straight-ish shot whose centre is at (x, z) when it is a ball-radius past the line."""
    t_cross = 0.05
    y_cross = GOAL.line_y + BALL_RADIUS
    v = np.array([vx, vy, 0.0])
    # choose vz so that z(t_cross) = z_cross starting from z = 1.0
    vz = (z_cross - 1.0 + 0.5 * G * t_cross**2) / t_cross
    v[2] = vz
    p0 = np.array([x_cross - vx * t_cross, y_cross - vy * t_cross, 1.0])
    seg = Segment.ballistic(0.0, 0.12, p0, v, (0, 0, -G))
    return BallisticModel((seg,), ground_z=BALL_RADIUS, bounce=None)


def test_shot_into_the_top_corner_is_a_goal():
    d = football.goal_line(shot(3.0, 2.0))
    assert d.label == "GOAL" and d.margin_mm > 0


def test_shot_wide_of_the_post_is_not_a_goal():
    assert football.goal_line(shot(4.2, 1.0)).label == "NO_GOAL"


def test_shot_over_the_bar_is_not_a_goal():
    assert football.goal_line(shot(0.0, 2.8)).label == "NO_GOAL"


def test_ball_must_be_wholly_inside_the_post():
    inner = GOAL.half_width - BALL_RADIUS
    assert football.goal_line(shot(inner - 0.005, 1.0)).label == "GOAL"
    assert football.goal_line(shot(inner + 0.005, 1.0)).label == "NO_GOAL"


def test_ball_must_be_wholly_under_the_bar():
    inner = GOAL.height - BALL_RADIUS
    assert football.goal_line(shot(0.0, inner - 0.005)).label == "GOAL"
    assert football.goal_line(shot(0.0, inner + 0.005)).label == "NO_GOAL"


def test_margin_is_distance_to_nearest_frame_edge():
    d = football.goal_line(shot(1.0, GOAL.height - BALL_RADIUS - 0.02))
    assert d.margin_mm == pytest.approx(20.0, abs=0.5)


def test_impact_is_the_crossing_point():
    d = football.goal_line(shot(-1.5, 0.8))
    assert d.impact == pytest.approx((-1.5, 0.8), abs=1e-6)


def test_ball_that_never_reaches_the_line_is_not_a_goal():
    seg = Segment.ballistic(0, 0.1, (0, -5.0, 0.5), (0, -3.0, 0), (0, 0, -G))
    model = BallisticModel((seg,), ground_z=BALL_RADIUS, bounce=None)
    d = football.goal_line(model)
    assert d.label == "NO_GOAL" and d.impact is None
