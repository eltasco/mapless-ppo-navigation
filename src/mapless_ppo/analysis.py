"""Summarize observed timeout signatures, without inventing missing trajectories.

Usage:
    python scripts/timeout_summary.py results/paired_benchmark.csv --obstacles 2
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def analyze(rows: list[dict], obstacles: int | None = None, controller: str = "ppo") -> dict:
    selected = [r for r in rows if r["controller"] == controller and
                (obstacles is None or int(r["obstacles"]) == obstacles)]
    if not selected:
        raise ValueError("No matching rows; supply a newly generated episode CSV")
    timeouts = [r for r in selected if r["outcome"] == "timeout"]
    def stats(name: str):
        values = [float(r[name]) for r in timeouts]
        return {"mean": round(statistics.mean(values), 4),
                "median": round(statistics.median(values), 4),
                "min": round(min(values), 4), "max": round(max(values), 4)} if values else None
    return {
        "controller": controller, "obstacles": obstacles,
        "total_episodes": len(selected), "timeout_episodes": len(timeouts),
        "timeout_seeds": [int(r["seed"]) for r in timeouts],
        "low_speed_fraction_among_timeouts": stats("low_speed_fraction"),
        "velocity_sign_flips_among_timeouts": stats("velocity_sign_flips"),
        "end_goal_distance_m_among_timeouts": stats("end_goal_distance_m"),
        "interpretation": "These are proxies for stalling/oscillation, not proofs of a specific cause.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Per-episode CSV produced by benchmark")
    ap.add_argument("--obstacles", type=int, default=2)
    ap.add_argument("--controller", default="ppo")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    with Path(args.csv).open(newline="", encoding="utf-8") as fp:
        summary = analyze(list(csv.DictReader(fp)), args.obstacles, args.controller)
    content = json.dumps(summary, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content + "\n", encoding="utf-8")
    print(content)


if __name__ == "__main__":
    main()
