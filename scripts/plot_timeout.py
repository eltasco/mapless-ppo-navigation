"""Plot up to a few saved timeout traces produced by `run.py benchmark`.

Requires matplotlib, included in the optional 'analysis' dependencies.
Example: python scripts/plot_timeout.py results/timeout_traces.csv results/timeouts.png
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="CSV exported by benchmark --trace-csv")
    parser.add_argument("output", type=Path, help="Output figure (.png)")
    args = parser.parse_args()
    from matplotlib import pyplot as plt
    traces = defaultdict(list)
    with args.input.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            key = (row["controller"], row["obstacles"], row["seed"])
            traces[key].append((float(row["x_m"]), float(row["y_m"])))
    if not traces:
        parser.error("No timeout traces in CSV; increase --episodes or --trace-timeouts")
    fig, ax = plt.subplots(figsize=(8, 6))
    for (controller, obstacles, seed), coords in traces.items():
        ax.plot([x for x, _ in coords], [y for _, y in coords], label=f"{controller} / {obstacles} obstacles / seed {seed}")
    ax.scatter([0], [0], marker="x", label="Start (0,0)")
    ax.set(xlabel="x (m)", ylabel="y (m)", title="Selected timeout trajectories")
    ax.axis("equal")
    ax.grid(True, alpha=.3)
    ax.legend(loc="best", fontsize="small")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
