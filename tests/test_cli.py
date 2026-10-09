import json
import shutil

import pytest
from PIL import Image

from parallax import cli

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def test_clip_then_analyze_round_trip(tmp_path, capsys):
    out = tmp_path / "clip.npz"
    assert cli.main(["clip", "--sport", "football", "--seed", "2", "--out", str(out)]) == 0
    capsys.readouterr()
    assert cli.main(["analyze", str(out)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["sport"] == "football"
    assert report["decision"] in {"GOAL", "NO_GOAL"}
    assert report["points"] > 20


@needs_ffmpeg
@pytest.mark.parametrize("sport", ["tennis", "cricket", "football"])
def test_demo_writes_a_small_animated_gif(tmp_path, sport):
    out = tmp_path / f"{sport}.gif"
    assert cli.main(["demo", "--sport", sport, "--out", str(out)]) == 0
    assert 0 < out.stat().st_size <= 4 * 1024 * 1024
    with Image.open(out) as gif:
        assert gif.format == "GIF" and gif.n_frames > 40
        assert gif.size[0] <= 900


def test_unknown_sport_is_rejected():
    with pytest.raises(SystemExit):
        cli.main(["demo", "--sport", "curling", "--out", "x.gif"])
