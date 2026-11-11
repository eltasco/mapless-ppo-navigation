"""Validate the recorded console-summary transcription, without rerunning PPO."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_reported_outcomes_are_consistent():
    with (ROOT / "results/paired_benchmark_summary.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    totals = defaultdict(lambda: {"episodes": 0, "successes": 0, "collisions": 0, "timeouts": 0})
    for row in rows:
        assert row["provenance"] == "user_supplied_local_benchmark_stdout_2026-11-11"
        n = int(row["episodes"])
        s = int(row["successes"])
        c = int(row["collisions"])
        t = int(row["timeouts"])
        assert n == s + c + t == 500
        assert float(row["success_rate_pct"]) == 100 * s / n
        lo = float(row["wilson95_low_pct_rounded"])
        hi = float(row["wilson95_high_pct_rounded"])
        assert lo <= 100 * s / n <= hi
        for key, value in (("episodes", n), ("successes", s), ("collisions", c), ("timeouts", t)):
            totals[row["controller"]][key] += value

    meta = json.loads((ROOT / "results/paired_benchmark_summary.json").read_text(encoding="utf-8"))
    assert meta["status"] == "user_reported_measured_results_not_independently_rerun"
    assert meta["results"] == [
        {k: (int(v) if k in {"obstacles", "episodes", "successes", "collisions", "timeouts"} else float(v) if k in {"success_rate_pct", "wilson95_low_pct_rounded", "wilson95_high_pct_rounded"} else v)
         for k, v in row.items()}
        for row in rows
    ]
    for name in ("ppo", "potential"):
        assert dict(totals[name]) == {k: meta["aggregate"][name][k] for k in totals[name]}
    assert totals["ppo"]["successes"] == 1817
    assert totals["potential"]["successes"] == 149
