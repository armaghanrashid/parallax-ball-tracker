import numpy as np
import pytest

from parallax.rules import tennis
from parallax.rules.tennis import BASELINE, FOOTPRINT_M, HALF_LINE_WIDTH_M, Line
from parallax.vision.fit import BallisticModel, Segment
from tests.helpers import bounce_model

OUTER_EDGE_Y = 11.885


def call_at(y, x=0.0, line=BASELINE):
    return tennis.call(bounce_model((x, y)), line)


def test_ball_well_inside_is_in():
    assert call_at(11.5).label == "IN"


def test_ball_well_outside_is_out():
    assert call_at(12.2).label == "OUT"


def test_ball_centred_on_the_line_is_in():
    assert call_at(OUTER_EDGE_Y - HALF_LINE_WIDTH_M).label == "IN"


def test_ball_touching_outer_edge_counts_as_in():
    d = call_at(OUTER_EDGE_Y)
    assert d.label == "IN"
    assert d.margin_mm == pytest.approx(FOOTPRINT_M * 1000)


def test_footprint_just_touching_the_line_is_in_with_zero_margin():
    d = call_at(OUTER_EDGE_Y + FOOTPRINT_M)
    assert d.label == "IN"
    assert d.margin_mm == pytest.approx(0.0, abs=1e-6)


def test_one_millimetre_past_the_footprint_is_out():
    d = call_at(OUTER_EDGE_Y + FOOTPRINT_M + 0.001)
    assert d.label == "OUT"
    assert d.margin_mm == pytest.approx(-1.0, abs=1e-6)


def test_ball_beyond_the_end_of_the_line_is_out():
    assert call_at(11.7, x=4.4).label == "OUT"  # wide of the singles sideline


def test_decision_reports_the_bounce_point():
    d = call_at(11.9, x=1.25)
    assert d.impact == pytest.approx((1.25, 11.9))


def test_sideline_uses_its_own_orientation():
    side = Line(p0=(4.115, -11.885), p1=(4.115, 11.885), half_width=0.025, inward=(-1.0, 0.0))
    inside = tennis.call(bounce_model((4.0, 3.0)), side)
    outside = tennis.call(bounce_model((4.2, 3.0)), side)
    assert inside.label == "IN" and outside.label == "OUT"


def test_no_bounce_means_no_call():
    seg = Segment.ballistic(0, 1, (0, 0, 1), (0, 10, 0), (0, 0, -9.8))
    model = BallisticModel((seg,), ground_z=0.0335, bounce=None)
    with pytest.raises(tennis.NoBounce):
        tennis.call(model, BASELINE)


def test_margin_sign_agrees_with_label():
    for y in np.linspace(11.6, 12.1, 26):
        d = call_at(float(y))
        assert (d.margin_mm >= 0) == (d.label == "IN")
