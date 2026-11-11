"""Validate actual per-episode benchmark CSV against its JSON aggregate.

This program *never fabricates* episodes from a console transcript. Use it
only on files written by `python run.py benchmark` with stable-baselines3.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def verify(csv_path: Path, json_path: Path) -> dict:
    raw = csv_path.read_bytes()
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    if not rows or not meta.get("summaries"):
        raise ValueError("Missing episode rows or summary metadata")
    counts = collections.Counter()
    seen = set()
    for row in rows:
        key = (row["controller"], row["stage"], int(row["obstacles"]), int(row["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate controller/configuration/seed: {key}")
        seen.add(key)
        if row["outcome"] not in ("success", "collision", "timeout"):
            raise ValueError(f"Invalid outcome {row['outcome']}")
        counts[key[:3] + (row["outcome"],)] += 1
    for item in meta["summaries"]:
        k = (item["controller"], item["stage"], int(item["obstacles"]))
        for field, name in [("success", "success"), ("collision", "collision"), ("timeout", "timeout")]:
            if counts[k + (name,)] != int(item[field]):
                raise ValueError(f"Count mismatch for {k}, {name}")
        if sum(counts[k + (name,)] for name in ("success", "collision", "timeout")) != int(item["episodes"]):
            raise ValueError(f"Missing episodes for {k}")
    return {"csv_sha256": hashlib.sha256(raw).hexdigest(),
            "json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
            "rows": len(rows), "configurations": len(meta["summaries"]),
            "status": "episode_artifacts_verified_against_aggregates"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("json")
    args = ap.parse_args()
    print(json.dumps(verify(Path(args.csv), Path(args.json)), indent=2))


if __name__ == "__main__":
    main()
