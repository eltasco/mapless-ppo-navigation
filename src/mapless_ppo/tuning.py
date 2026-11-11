"""Grid-search classical-controller gains on *validation* seeds, never test seeds."""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

from .baselines import PotentialFieldController
from .benchmark import make_benchmark_env
from .rollouts import run_episode


def candidate_parameters():
    """Deterministic limited grid including turning and reverse variants."""
    reference = PotentialFieldController()
    for repulsion, tangent_gain, reverse in itertools.product(
            (0.08, 0.16, 0.32), (0.0, 0.45, 0.9), (False, True)):
        yield replace(reference, repulsion=repulsion,
                      tangent_gain=tangent_gain, allow_reverse=reverse)


def validate_seed_split(seed_start: int, episodes: int):
    if seed_start < 0 or episodes < 1:
        raise ValueError("Invalid validation range")
    # Historical held-out range: 60000-60499. Never tune on these seeds.
    if not (seed_start + episodes - 1 < 60_000 or seed_start > 60_499):
        raise ValueError("Validation seed range overlaps the published 60000–60499 test range")


def tune_baseline(*, stage: str = "hard", counts: list[int] | None = None,
                  seed_start: int = 30_000, episodes: int = 30,
                  output: str = "configs/potential_validated.json", force: bool = False):
    counts = counts or [2, 3, 4, 5]
    validate_seed_split(seed_start, episodes)
    if not counts or min(counts) < 1:
        raise ValueError("Obstacle counts must be positive")
    target = Path(output)
    if target.exists() and not force:
        raise FileExistsError(f"Will not replace existing tuned configuration: {target}")
    records = []
    for i, candidate in enumerate(candidate_parameters(), 1):
        aggregate = Counter()
        steps = 0
        for n in counts:
            env = make_benchmark_env(n, stage, "original", 0.0, 0.0, "single")
            try:
                for j in range(episodes):
                    metrics, _ = run_episode(env, candidate, seed_start + j, None)
                    aggregate[metrics.outcome] += 1
                    steps += metrics.steps
            finally:
                env.close()
        total = len(counts) * episodes
        # Success is primary; collisions are penalized separately. A policy
        # that never moves receives a poor score via its low success fraction.
        score = ((aggregate["success"] - 0.5 * aggregate["collision"])
                 / total - 0.02 * steps / (total * 900))
        record = {
            "parameters": asdict(candidate), "success": aggregate["success"],
            "collision": aggregate["collision"], "timeout": aggregate["timeout"],
            "episodes": total, "score": round(float(score), 8),
        }
        records.append(record)
        print(f"{i:02d}: success {aggregate['success']}/{total}, collisions "
              f"{aggregate['collision']}, timeouts {aggregate['timeout']} | score={score:.4f}", flush=True)
    selected = max(records, key=lambda r: (r["score"], r["success"], -r["collision"]))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "validated_potential_field_v2", "stage": stage,
        "obstacles": counts, "validation_seed_start": seed_start,
        "episodes_per_obstacle_count": episodes,
        "test_seed_exclusion": "60000–60499 excluded from tuning",
        "objective": "success_rate - 0.5 * collision_rate - 0.02 * normalized_steps",
        "parameters": selected["parameters"], "best_validation_metrics": selected,
        "all_candidates": records,
        "note": "Tuning results are validation-only, not held-out performance."
    }
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved validation-selected baseline: {target}")
    return payload
