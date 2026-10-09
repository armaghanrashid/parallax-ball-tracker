import numpy as np
import pytest

from parallax.camera import Camera


@pytest.fixture
def rig() -> tuple[Camera, Camera]:
    """Two cameras looking at the origin from different sides."""
    a = Camera.look_at((-8.0, 12.0, 4.0), (0.0, 0.0, 0.5), focal_px=1100.0, size=(640, 360))
    b = Camera.look_at((9.0, 11.0, 5.0), (0.0, 0.0, 0.5), focal_px=1100.0, size=(640, 360))
    return a, b


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
