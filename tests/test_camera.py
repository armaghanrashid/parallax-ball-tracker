import numpy as np

from parallax.camera import Camera


def test_look_at_points_target_to_image_centre():
    cam = Camera.look_at((3.0, -10.0, 2.0), (1.0, 2.0, 0.5), focal_px=900.0, size=(640, 360))
    uv = cam.project(np.array([1.0, 2.0, 0.5]))
    assert np.allclose(uv, [319.5, 179.5], atol=1e-9)


def test_rotation_is_orthonormal_and_right_handed():
    cam = Camera.look_at((5.0, 5.0, 3.0), (0.0, 0.0, 0.0), focal_px=800.0, size=(640, 360))
    assert np.allclose(cam.R @ cam.R.T, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(cam.R), 1.0)


def test_camera_centre_round_trips():
    pos = np.array([-4.0, 7.0, 2.5])
    cam = Camera.look_at(pos, (0.0, 0.0, 0.0), focal_px=800.0, size=(640, 360))
    assert np.allclose(cam.center, pos)


def test_world_up_is_image_up():
    cam = Camera.look_at((0.0, -10.0, 1.0), (0.0, 0.0, 1.0), focal_px=800.0, size=(640, 360))
    low = cam.project(np.array([0.0, 0.0, 1.0]))
    high = cam.project(np.array([0.0, 0.0, 2.0]))
    assert high[1] < low[1]  # image y grows downward


def test_projection_matrix_matches_project(rig):
    cam = rig[0]
    X = np.array([[0.3, -0.2, 1.1], [1.0, 2.0, 0.4]])
    h = np.c_[X, np.ones(len(X))] @ cam.P.T
    assert np.allclose(h[:, :2] / h[:, 2:3], cam.project(X))


def test_array_round_trip(rig):
    cam = rig[1]
    restored = Camera.from_arrays(*cam.to_arrays())
    assert np.allclose(restored.P, cam.P)
    assert (restored.width, restored.height) == (cam.width, cam.height)


def test_depth_positive_in_front(rig):
    assert rig[0].depth(np.array([0.0, 0.0, 0.5])) > 0
