"""Command line: render a demo GIF, write a clip for the Lambda handler, analyse a clip.

python -m parallax.cli demo --sport tennis --out docs/media/tennis.gif
python -m parallax.cli clip --sport cricket --seed 7 --out clip.npz
python -m parallax.cli analyze clip.npz
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

from parallax.clip import Clip, load_clip, save_clip
from parallax.pipeline import analyze
from parallax.sim import simulate
from parallax.sim.render import render
from parallax.sports import SPORTS

# Seeds chosen for readable demos: a close call that the pipeline gets right.
DEMO_SEEDS = {"tennis": 3, "cricket": 6, "football": 1}


def _simulate_and_render(sport: str, seed: int):
    delivery = simulate(sport, seed)
    frames = [render(delivery, cam) for cam in delivery.cameras]
    return delivery, frames


def cmd_demo(args) -> int:
    from parallax.viz import build_frames, write_gif

    seed = DEMO_SEEDS[args.sport] if args.seed is None else args.seed
    delivery, frames = _simulate_and_render(args.sport, seed)
    result = analyze(frames, delivery.cameras, args.sport)
    gif = build_frames(delivery, frames, result)
    size = write_gif(gif, args.out)
    print(
        f"{args.sport} seed {seed}: {result.decision.label} "
        f"(truth {delivery.truth.label}), {len(gif)} frames, {size / 1e6:.2f} MB -> {args.out}"
    )
    return 0


def cmd_clip(args) -> int:
    seed = DEMO_SEEDS[args.sport] if args.seed is None else args.seed
    delivery, frames = _simulate_and_render(args.sport, seed)
    save_clip(args.out, Clip(np.stack(frames), delivery.cameras, args.sport))
    print(f"wrote {args.out} ({args.sport}, seed {seed}, true call {delivery.truth.label})")
    return 0


def cmd_analyze(args) -> int:
    clip = load_clip(args.clip)
    result = analyze(clip.frames, clip.cameras, clip.sport, clip.fps)
    d = result.decision
    print(
        json.dumps(
            {
                "sport": clip.sport,
                "decision": d.label,
                "margin_mm": None if d.margin_mm is None else round(d.margin_mm, 2),
                "impact": d.impact,
                "points": int(len(result.points)),
                "latency_ms": round(result.latency_ms, 1),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m parallax.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="render a third-umpire GIF")
    demo.add_argument("--sport", choices=sorted(SPORTS), required=True)
    demo.add_argument("--out", required=True)
    demo.add_argument("--seed", type=int)
    demo.set_defaults(func=cmd_demo)

    clip = sub.add_parser("clip", help="write a simulated clip (.npz) for the Lambda handler")
    clip.add_argument("--sport", choices=sorted(SPORTS), required=True)
    clip.add_argument("--out", required=True)
    clip.add_argument("--seed", type=int)
    clip.set_defaults(func=cmd_clip)

    an = sub.add_parser("analyze", help="analyse a clip file and print the decision as JSON")
    an.add_argument("clip")
    an.set_defaults(func=cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
