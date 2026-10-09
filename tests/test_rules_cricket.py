import numpy as np
import pytest

from parallax.rules import cricket
from parallax.rules.cricket import BALL_RADIUS, STUMPS
from parallax.vision.fit import BallisticModel, Segment

G = 9.80665


def post_bounce_model(x_at_stumps, z_at_stumps, vy=28.0, ground_z=BALL_RADIUS, x_drift=0.0):
    """Observed arc that ends 1 m before the stumps; the rule extrapolates it."""
    t_plane = 0.2
    y_plane = STUMPS.plane_y
    vz = (z_at_stumps - 0.9 + 0.5 * G * t_plane**2) / t_plane
    p0 = np.array([x_at_stumps - x_drift * t_plane, y_plane - vy * t_plane, 0.9])
    v0 = np.array([x_drift, vy, vz])
    seg = Segment.ballistic(0.0, t_plane - 1.0 / vy, p0, v0, (0, 0, -G))
    return BallisticModel((seg,), ground_z=ground_z, bounce=None)


def test_ball_through_the_middle_of_the_stumps_hits():
    d = cricket.lbw(post_bounce_model(0.0, 0.35))
    assert d.label == "HIT"
    assert d.impact == pytest.approx((0.0, 0.35), abs=1e-6)
    assert d.margin_mm > 0


def test_ball_wide_of_leg_stump_misses():
    assert cricket.lbw(post_bounce_model(0.30, 0.35)).label == "MISS"


def test_ball_above_the_bails_misses():
    top = STUMPS.height + BALL_RADIUS
    assert cricket.lbw(post_bounce_model(0.0, top + 0.03)).label == "MISS"


def test_ball_clipping_the_bails_still_hits():
    top = STUMPS.height + BALL_RADIUS
    d = cricket.lbw(post_bounce_model(0.0, top - 0.01))
    assert d.label == "HIT"
    assert d.margin_mm == pytest.approx(10.0, abs=0.5)


def test_ball_clipping_the_outside_edge_hits_using_ball_radius():
    edge = STUMPS.half_width + BALL_RADIUS
    assert cricket.lbw(post_bounce_model(edge - 0.005, 0.3)).label == "HIT"
    assert cricket.lbw(post_bounce_model(edge + 0.005, 0.3)).label == "MISS"


def test_off_centre_stumps_are_respected():
    stumps = cricket.Stumps(plane_y=STUMPS.plane_y, x_center=0.5)
    assert cricket.lbw(post_bounce_model(0.5, 0.3), stumps).label == "HIT"
    assert cricket.lbw(post_bounce_model(0.0, 0.3), stumps).label == "MISS"


def test_lateral_drift_is_projected_to_the_stump_plane():
    d = cricket.lbw(post_bounce_model(0.05, 0.3, x_drift=0.8))
    assert d.impact[0] == pytest.approx(0.05, abs=1e-6)


def test_ball_that_never_reaches_the_stumps_is_a_miss():
    seg = Segment.ballistic(0, 0.1, (0, 5.0, 1.0), (0, -10.0, 0), (0, 0, -G))
    model = BallisticModel((seg,), ground_z=BALL_RADIUS, bounce=None)
    d = cricket.lbw(model)
    assert d.label == "MISS" and d.impact is None
