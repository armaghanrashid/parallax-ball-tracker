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
        est = np.array([result.model.position(i / 240.0) for i in range(len(truth))])
        rmse_fit = float(np.sqrt(((est - truth) ** 2).sum(axis=1).mean())) * 1000.0

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


def run(n: int = 200, sports: list[str] | None = None, seed0: int = 0, workers: int = 1) -> dict:
    """Evaluate `n` clips per sport (seeds seed0 .. seed0+n-1). Returns metrics per sport."""
    out = {}
    for sport in sports or list(SPORTS):
        jobs = [(sport, seed0 + i) for i in range(n)]
        if workers > 1:
            with mp.Pool(workers) as pool:
                rows = pool.starmap(evaluate_clip, jobs)
        else:
            rows = [evaluate_clip(*job) for job in jobs]
        out[sport] = summarise(rows)
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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m parallax.eval", description=__doc__)
    parser.add_argument("--n", type=int, default=200, help="clips per sport")
    parser.add_argument("--seed", type=int, default=0, help="first seed")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--json", help="also write the raw metrics to this file")
    args = parser.parse_args(argv)
    results = run(args.n, seed0=args.seed, workers=args.workers)
    print(format_table(results))
    print(
        f"\nClose call: |true margin| < {CLOSE_CALL_MM:.0f} mm. "
        "Latency is analysis only (detect, track, triangulate, fit, rule), single process."
    )
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
