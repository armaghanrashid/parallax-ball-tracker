import numpy as np
import pytest

from parallax.pipeline import NO_DECISION, analyze
from parallax.sim import simulate
from parallax.sim.render import render

SPORTS = ["tennis", "cricket", "football"]


@pytest.fixture(scope="module", params=SPORTS)
def clip(request):
    sport = request.param
    d = simulate(sport, 4)
    frames = [render(d, cam) for cam in d.cameras]
    return d, frames, analyze(frames, d.cameras, sport)


def test_reconstruction_is_centimetre_accurate(clip):
    d, _, result = clip
    truth = d.visible_pos[result.frame_index]
    err = np.linalg.norm(result.points[:, 1:] - truth, axis=1)
    assert len(err) > 0.8 * len(d.visible_pos)
    assert np.sqrt((err**2).mean()) < 0.015


def test_fitted_model_follows_the_true_path(clip):
    d, _, result = clip
    t = d.visible_t - d.visible_t[0]
    est = np.array([result.model.position(ti) for ti in t])
    assert np.sqrt(((est - d.visible_pos) ** 2).sum(axis=1).mean()) < 0.015


def test_decision_matches_the_ground_truth_when_not_borderline(clip):
    d, _, result = clip
    if abs(d.truth.margin_mm) > 40:
        assert result.decision.label == d.truth.label
    assert result.latency_ms > 0


def test_margin_error_is_small(clip):
    d, _, result = clip
    assert abs(result.decision.margin_mm - d.truth.margin_mm) < 40


def test_footage_without_a_ball_gives_no_decision():
    d = simulate("tennis", 1)
    blank = [[np.full((360, 640, 3), 30, np.uint8)] * 20 for _ in range(2)]
    result = analyze(blank, d.cameras, "tennis")
    assert result.decision.label == NO_DECISION
    assert result.model is None


def test_unknown_sport_rejected():
    d = simulate("tennis", 1)
    with pytest.raises(KeyError):
        analyze([[], []], d.cameras, "curling")


def test_accepts_stacked_arrays():
    d = simulate("football", 2)
    frames = np.stack([np.stack(render(d, cam)) for cam in d.cameras])
    result = analyze(frames, d.cameras, "football")
    assert result.decision.label in {"GOAL", "NO_GOAL"}
