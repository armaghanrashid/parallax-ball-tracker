import json

from parallax import eval as ev


def test_run_reports_metrics_for_every_sport():
    res = ev.run(n=2)
    assert set(res) == {"tennis", "cricket", "football"}
    for stats in res.values():
        assert stats["n"] == 2
        for key in ("rmse_mm", "rmse_fit_mm", "accuracy_pct", "latency_ms", "coverage_pct"):
            assert key in stats and stats[key] >= 0
        assert stats["rmse_mm"] < 50


def test_run_is_deterministic_apart_from_latency():
    a, b = ev.run(n=2, sports=["football"]), ev.run(n=2, sports=["football"])
    a["football"].pop("latency_ms"), b["football"].pop("latency_ms")
    assert json.dumps(a) == json.dumps(b)  # json: NaN-safe equality


def test_table_is_markdown_with_one_row_per_sport():
    table = ev.format_table(ev.run(n=1))
    lines = table.splitlines()
    assert lines[0].startswith("| Sport") and set(lines[1]) <= set("|- :")
    assert len(lines) == 5
    assert all("|" in line for line in lines)


def test_main_prints_table_and_can_write_json(tmp_path, capsys):
    out = tmp_path / "metrics.json"
    plot = tmp_path / "margins.png"
    ev.main(["--n", "1", "--json", str(out), "--plot", str(plot)])
    assert plot.stat().st_size > 1000
    assert "| tennis" in capsys.readouterr().out
    assert json.loads(out.read_text())["tennis"]["n"] == 1
