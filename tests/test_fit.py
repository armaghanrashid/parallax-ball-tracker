import numpy as np
import pytest

from parallax.vision.fit import BallisticModel, InsufficientData, Segment, fit_trajectory
from tests.helpers import bounce_samples, parabola_points


def test_segment_evaluates_polynomial():
    seg = Segment.ballistic(0.0, 2.0, (1, 2, 3), (4, 5, 6), (0, 0, -10))
    assert np.allclose(seg.position(0.5), [3.0, 4.5, 3 + 3 - 1.25])


def test_noiseless_bounce_recovers_contact_point_within_a_millimetre():
    pts, truth = bounce_samples(noise=0.0)
    model = fit_trajectory(pts, ground_z=0.0335)
    assert model.bounce is not None
    assert np.linalg.norm(model.bounce_point() - truth["pb"]) < 1e-3
    assert abs(model.bounce[0] - truth["tb"]) < 1e-3
    assert len(model.segments) == 2


def test_noisy_bounce_contact_point_within_a_centimetre():
    errors = []
    for seed in range(10):
        pts, truth = bounce_samples(noise=0.002, seed=seed)
        model = fit_trajectory(pts, ground_z=0.0335)
        errors.append(np.linalg.norm(model.bounce_point() - truth["pb"]))
    assert max(errors) < 0.01


def test_positions_follow_both_arcs():
    pts, _ = bounce_samples(noise=0.002, seed=3)
    model = fit_trajectory(pts, ground_z=0.0335)
    rows = zip(pts[:, 0], pts[:, 1:], strict=True)
    err = np.array([np.linalg.norm(model.position(t) - p) for t, p in rows])
    assert err.max() < 0.01


def test_single_arc_without_bounce_has_one_segment():
    t = np.arange(30) / 240.0
    pts = np.c_[t, parabola_points(t, (0, 0, 1.0), (2, 25, 3))]
    model = fit_trajectory(pts, ground_z=0.11)
    assert model.bounce is None and model.bounce_point() is None
    assert len(model.segments) == 1


def test_single_outlier_does_not_move_the_fit():
    pts, truth = bounce_samples(noise=0.001, seed=2)
    pts[7, 1:] += np.array([0.4, -0.3, 0.2])
    model = fit_trajectory(pts, ground_z=0.0335)
    assert np.linalg.norm(model.bounce_point() - truth["pb"]) < 0.01


def test_cross_y_inside_and_extrapolated():
    t = np.arange(30) / 240.0
    pts = np.c_[t, parabola_points(t, (0, 0, 1.0), (2, 25, 3))]
    model = fit_trajectory(pts, ground_z=0.11)
    tc, xyz = model.cross_y(1.0)  # inside the observed span
    assert np.isclose(xyz[1], 1.0, atol=1e-6) and np.isclose(tc, 0.04, atol=1e-6)
    tc, xyz = model.cross_y(10.0)  # extrapolated beyond the data
    assert np.isclose(xyz[1], 10.0, atol=1e-6) and tc > t[-1]
    assert model.cross_y(-5.0) is None  # behind the start: never crossed going forwards


def test_too_few_points_rejected():
    with pytest.raises(InsufficientData):
        fit_trajectory(np.zeros((3, 4)), ground_z=0.03)


def test_model_type():
    pts, _ = bounce_samples()
    assert isinstance(fit_trajectory(pts, ground_z=0.0335), BallisticModel)
