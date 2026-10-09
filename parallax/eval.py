"""Benchmark the pipeline on simulated clips against exact ground truth.

python -m parallax.eval --n 200
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys

import numpy as np

from parallax.pipeline import analyze
from parallax.sim import simulate
from parallax.sim.render import render
from parallax.sports import SPORTS

CLOSE_CALL_MM = 10.0  # |true margin| below this counts as a close call


def evaluate_clip(sport: str, seed: int) -> dict:
    """Simulate, render, analyse one clip and score it against the ground truth."""
    d = simulate(sport, seed)
    frames = [render(d, cam) for cam in d.cameras]
    result = analyze(frames, d.cameras, sport)

    truth = d.visible_pos
    rmse_raw = rmse_fit = float("nan")
    if len(result.points):
        err = np.linalg.norm(result.points[:, 1:] - truth[result.frame_index], axis=1)
        rmse_raw = float(np.sqrt((err**2).mean())) * 1000.0
    if result.model is not None:
        # Score the model only where it has data; frames outside would be pure extrapolation.
        frames = np.arange(result.frame_index.min(), result.frame_index.max() + 1)
        est = np.array([result.model.position(i / 240.0) for i in frames])
        rmse_fit = float(np.sqrt(((est - truth[frames]) ** 2).sum(axis=1).mean())) * 1000.0

    margin_err = float("nan")
    if result.decision.margin_mm is not None and d.truth.margin_mm is not None:
        margin_err = abs(result.decision.margin_mm - d.truth.margin_mm)
    return {
        "correct": result.decision.label == d.truth.label,
        "decided": result.model is not None and result.decision.label != "NO_DECISION",
        "close": d.truth.margin_mm is not None and abs(d.truth.margin_mm) < CLOSE_CALL_MM,
        "rmse_raw": rmse_raw,
        "rmse_fit": rmse_fit,
        "margin_err": margin_err,
        "margin_true": d.truth.margin_mm,
        "margin_est": result.decision.margin_mm,
        "latency_ms": result.latency_ms,
    }


def _rms(values: list[float]) -> float:
    v = np.array([x for x in values if not np.isnan(x)])
    return float(np.sqrt((v**2).mean())) if len(v) else float("nan")


def summarise(rows: list[dict]) -> dict:
    close = [r for r in rows if r["close"]]
    n = len(rows)
    return {
        "n": n,
        "coverage_pct": 100.0 * sum(r["decided"] for r in rows) / n,
        "rmse_mm": _rms([r["rmse_raw"] for r in rows]),
        "rmse_fit_mm": _rms([r["rmse_fit"] for r in rows]),
        "accuracy_pct": 100.0 * sum(r["correct"] for r in rows) / n,
        "close_n": len(close),
        "close_accuracy_pct": (
            100.0 * sum(r["correct"] for r in close) / len(close) if close else float("nan")
        ),
        "margin_mae_mm": float(np.nanmean([r["margin_err"] for r in rows])),
        "latency_ms": float(np.mean([r["latency_ms"] for r in rows])),
    }


def run(
    n: int = 200,
    sports: list[str] | None = None,
    seed0: int = 0,
    workers: int = 1,
    keep_rows: dict | None = None,
) -> dict:
    """Evaluate `n` clips per sport (seeds seed0 .. seed0+n-1). Returns metrics per sport.

    If `keep_rows` is a dict it is filled with the per-clip rows, keyed by sport.
    """
    out = {}
    for sport in sports or list(SPORTS):
        jobs = [(sport, seed0 + i) for i in range(n)]
        if workers > 1:
            with mp.Pool(workers) as pool:
                rows = pool.starmap(evaluate_clip, jobs)
        else:
            rows = [evaluate_clip(*job) for job in jobs]
        out[sport] = summarise(rows)
        if keep_rows is not None:
            keep_rows[sport] = rows
    return out


def format_table(results: dict) -> str:
    head = (
        "| Sport | Clips | Decided (%) | 3D RMSE, triangulated (mm) | 3D RMSE, fitted (mm) "
        "| Call accuracy (%) | Close calls (n) | Close-call accuracy (%) | Mean latency (ms) |"
    )
    sep = "|:--|--:|--:|--:|--:|--:|--:|--:|--:|"
    rows = [head, sep]
    for sport, s in results.items():
        close = "n/a" if np.isnan(s["close_accuracy_pct"]) else f"{s['close_accuracy_pct']:.1f}"
        rows.append(
            f"| {sport} | {s['n']} | {s['coverage_pct']:.1f} | {s['rmse_mm']:.2f} "
            f"| {s['rmse_fit_mm']:.2f} | {s['accuracy_pct']:.1f} | {s['close_n']} | {close} "
            f"| {s['latency_ms']:.1f} |"
        )
    return "\n".join(rows)


def plot_margins(rows: dict, path: str) -> None:
    """Scatter of estimated vs true decision margin per sport (dark theme, 1600 px wide)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(1, len(rows), figsize=(16, 5.2), dpi=100, facecolor="#0d0f13")
    for ax, (sport, sport_rows) in zip(axes, rows.items(), strict=True):
        pts = np.array(
            [(r["margin_true"], r["margin_est"]) for r in sport_rows if r["margin_est"] is not None]
        )
        ax.set_facecolor("#161a20")
        lim = float(np.abs(pts).max()) if len(pts) else 1.0
        ax.plot([-lim, lim], [-lim, lim], color="#8a96a6", lw=1, ls="--", label="perfect")
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color="#40c9b0", alpha=0.8, label="clips")
        ax.axhline(0, color="#3a4452", lw=0.8)
        ax.axvline(0, color="#3a4452", lw=0.8)
        ax.set_title(sport, color="#e8ecf0", fontsize=13)
        ax.set_xlabel("true margin (mm)", color="#8a96a6")
        ax.set_ylabel("estimated margin (mm)", color="#8a96a6")
        ax.tick_params(colors="#8a96a6")
        for spine in ax.spines.values():
            spine.set_color("#3a4452")
    axes[0].legend(facecolor="#161a20", edgecolor="#3a4452", labelcolor="#e8ecf0")
    fig.suptitle(
        "Decision margin: reconstructed vs ground truth (positive = IN / HIT / GOAL)",
        color="#e8ecf0",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    Image.open(path).convert("RGB").save(path, optimize=True)  # re-save: drops any metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m parallax.eval", description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="clips per sport")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", help="also write the raw metrics to this file")
    parser.add_argument("--plot", help="also write a margin scatter plot (PNG) to this file")
    args = parser.parse_args(argv)
    rows: dict = {}
    results = run(args.n, seed0=args.seed, workers=args.workers, keep_rows=rows)
    print(format_table(results))
    print(
        f"\nClose call: |true margin| < {CLOSE_CALL_MM:.0f} mm. "
        "Latency is analysis only (detect, track, triangulate, fit, rule), single process."
    )
    if args.plot:
        plot_margins(rows, args.plot)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
