import numpy as np

from parallax.vision.triangulate import reprojection_error, triangulate, triangulate_many


def test_recovers_known_point_exactly(rig):
    c1, c2 = rig
    X = np.array([0.7, -0.4, 1.3])
    est = triangulate(c1.project(X), c2.project(X), c1.P, c2.P)
    assert np.linalg.norm(est - X) < 1e-6  # well under 1 mm without noise


def test_recovers_many_points_under_one_millimetre(rig, rng):
    c1, c2 = rig
    X = rng.uniform([-3, -3, 0.1], [3, 3, 3], size=(200, 3))
    est = triangulate_many(c1.project(X), c2.project(X), c1.P, c2.P)
    assert np.abs(est - X).max() < 1e-3


def test_batch_matches_single(rig, rng):
    c1, c2 = rig
    X = rng.uniform([-2, -2, 0.1], [2, 2, 2], size=(5, 3))
    p1, p2 = c1.project(X), c2.project(X)
    batch = triangulate_many(p1, p2, c1.P, c2.P)
    for i in range(5):
        assert np.allclose(batch[i], triangulate(p1[i], p2[i], c1.P, c2.P), atol=1e-9)


def test_noisy_pixels_give_millimetre_scale_error(rig, rng):
    c1, c2 = rig
    X = rng.uniform([-2, -2, 0.1], [2, 2, 2], size=(300, 3))
    n1 = c1.project(X) + rng.normal(0, 0.1, (300, 2))
    n2 = c2.project(X) + rng.normal(0, 0.1, (300, 2))
    err = np.linalg.norm(triangulate_many(n1, n2, c1.P, c2.P) - X, axis=1)
    assert np.sqrt((err**2).mean()) < 0.01  # 0.1 px noise -> well under 1 cm


def test_reprojection_error_zero_for_exact_and_large_for_inconsistent(rig):
    c1, c2 = rig
    X = np.array([[0.2, 0.1, 1.0]])
    p1, p2 = c1.project(X), c2.project(X)
    assert reprojection_error(X, p1, p2, c1.P, c2.P)[0] < 1e-6
    assert reprojection_error(X, p1 + 6.0, p2, c1.P, c2.P)[0] > 5.0
