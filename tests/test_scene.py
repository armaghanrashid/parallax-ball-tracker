import numpy as np
import pytest

from parallax.sim import Delivery, simulate
from parallax.sim.physics import SPECS

SPORTS = ["tennis", "cricket", "football"]


@pytest.mark.parametrize("sport", SPORTS)
def test_simulation_is_deterministic(sport):
    a, b = simulate(sport, 5), simulate(sport, 5)
    assert np.array_equal(a.flight.pos, b.flight.pos)
    assert not np.array_equal(a.flight.pos, simulate(sport, 6).flight.pos)


@pytest.mark.parametrize("sport", SPORTS)
def test_delivery_shape_and_metadata(sport):
    d = simulate(sport, 1)
    assert isinstance(d, Delivery)
    assert d.sport == sport and d.seed == 1
    assert d.ball_radius == SPECS[sport].radius
    assert 25 <= len(d.visible_pos) <= 160
    assert d.visible_pos.shape[1] == 3
    assert len(d.cameras) == 2
    assert np.allclose(np.diff(d.flight.t), 1 / 240)


@pytest.mark.parametrize("sport", SPORTS)
def test_ball_is_in_frame_for_both_cameras_throughout(sport):
    for seed in range(25):
        d = simulate(sport, seed)
        for cam in d.cameras:
            assert cam.in_frame(d.visible_pos, margin=6.0).all(), (sport, seed)


@pytest.mark.parametrize("sport", SPORTS)
def test_ball_is_a_few_pixels_wide(sport):
    d = simulate(sport, 0)
    for cam in d.cameras:
        r_px = cam.focal * d.ball_radius / cam.depth(d.visible_pos)
        assert r_px.min() > 1.8 and r_px.max() < 14


def test_both_tennis_calls_occur_and_bounce_is_visible():
    labels = set()
    for seed in range(40):
        d = simulate("tennis", seed)
        labels.add(d.truth.label)
        tb = d.flight.bounces[0].t
        assert d.visible_t[0] < tb < d.visible_t[-1]
    assert labels == {"IN", "OUT"}


def test_both_cricket_calls_occur_and_flight_reaches_the_stumps():
    labels = set()
    for seed in range(40):
        d = simulate("cricket", seed)
        labels.add(d.truth.label)
        assert d.flight.pos[-1, 1] >= 20.12  # truth runs on past the last observed frame
        assert d.visible_pos[-1, 1] < 20.12 - 0.5
        assert d.visible_t[0] < d.flight.bounces[0].t < d.visible_t[-1]
    assert labels == {"HIT", "MISS"}


def test_both_football_calls_occur_without_a_bounce():
    labels = set()
    for seed in range(40):
        d = simulate("football", seed)
        labels.add(d.truth.label)
        assert not d.flight.bounces
        assert d.visible_pos[:, 1].min() < -3 and d.visible_pos[:, 1].max() > 0
    assert labels == {"GOAL", "NO_GOAL"}


def test_unknown_sport_rejected():
    with pytest.raises(KeyError):
        simulate("curling", 0)
