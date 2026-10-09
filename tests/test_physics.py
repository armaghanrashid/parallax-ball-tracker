import numpy as np
import pytest

from parallax.sim.physics import SPECS, BallSpec, ballistic_velocity, integrate

G = 9.80665
DRAGLESS = BallSpec(radius=0.05, mass=0.1, cd=0.0, restitution=0.5, retention=0.8)


def test_dragless_flight_is_exactly_parabolic():
    fl = integrate(DRAGLESS, (0, 0, 2.0), (3.0, 10.0, 4.0), stop_y=8.0)
    t = fl.t
    expected = np.array([3.0 * t, 10.0 * t, 2.0 + 4.0 * t - 0.5 * G * t**2]).T
    assert np.abs(fl.pos - expected).max() < 1e-6


def test_samples_are_at_240_hz():
    fl = integrate(DRAGLESS, (0, 0, 2.0), (0, 10.0, 0.0), stop_y=5.0)
    assert np.allclose(np.diff(fl.t), 1 / 240)
    assert fl.t[0] == 0.0


def test_stops_after_reaching_stop_y():
    fl = integrate(DRAGLESS, (0, 0, 2.0), (0, 10.0, 0.0), stop_y=5.0)
    assert fl.pos[-1, 1] >= 5.0 > fl.pos[-2, 1]


def test_drag_slows_the_ball_and_bends_the_path():
    tennis = SPECS["tennis"]
    fl = integrate(tennis, (0, 0, 1.0), (0.0, 40.0, 3.0), stop_y=15.0)
    speed = np.linalg.norm(fl.vel, axis=1)
    assert speed[len(speed) // 2] < 41.0
    dragless = integrate(DRAGLESS, (0, 0, 1.0), (0.0, 40.0, 3.0), stop_y=15.0)
    assert fl.t[-1] > dragless.t[-1]  # slower to cover the same distance


def test_bounce_reflects_vertical_velocity_with_restitution():
    fl = integrate(DRAGLESS, (0, 0, 1.0), (0.0, 6.0, 0.0), stop_y=6.0)
    assert len(fl.bounces) >= 1
    b = fl.bounces[0]
    assert b.v_out[2] == pytest.approx(-DRAGLESS.restitution * b.v_in[2])
    assert b.v_out[1] == pytest.approx(DRAGLESS.retention * b.v_in[1])


def test_bounce_happens_when_the_centre_is_one_radius_above_ground():
    fl = integrate(DRAGLESS, (0, 0, 1.0), (0.0, 6.0, 0.0), stop_y=6.0)
    b = fl.bounces[0]
    tb = np.sqrt(2 * (1.0 - DRAGLESS.radius) / G)
    assert b.t == pytest.approx(tb, abs=1e-4)
    assert b.point[2] == 0.0  # contact point is on the ground plane
    assert b.point[1] == pytest.approx(6.0 * tb, abs=1e-3)


def test_ball_never_goes_through_the_ground():
    fl = integrate(SPECS["cricket"], (0, 0, 2.0), (0.0, 30.0, -3.0), stop_y=18.0)
    assert fl.pos[:, 2].min() >= SPECS["cricket"].radius - 1e-3


def test_custom_bounce_hook_is_applied():
    def kick(v):
        return (v[0] + 2.0, v[1], -0.5 * v[2])

    fl = integrate(DRAGLESS, (0, 0, 1.0), (0.0, 6.0, 0.0), stop_y=6.0, bounce_fn=kick)
    assert fl.bounces[0].v_out[0] == pytest.approx(2.0)


def test_ballistic_velocity_hits_the_aim_point_without_drag():
    p0, aim = np.array([0.0, 0.0, 1.0]), np.array([2.0, 12.0, 0.5])
    v0 = ballistic_velocity(p0, aim, speed=30.0)
    assert np.linalg.norm(v0) == pytest.approx(30.0)
    fl = integrate(DRAGLESS, p0, v0, stop_y=12.0)
    # interpolate to y = 12 and compare
    i = np.searchsorted(fl.pos[:, 1], 12.0)
    w = (12.0 - fl.pos[i - 1, 1]) / (fl.pos[i, 1] - fl.pos[i - 1, 1])
    hit = fl.pos[i - 1] + w * (fl.pos[i] - fl.pos[i - 1])
    assert np.abs(hit - aim).max() < 5e-3


def test_unreachable_aim_raises():
    with pytest.raises(ValueError):
        ballistic_velocity(np.array([0.0, 0.0, 0.0]), np.array([0.0, 100.0, 0.0]), speed=5.0)
