"""Seeded multi-scenario model selection for newly trained PPO policies.

Evaluates 2–5 obstacles (or a selected range) on fixed validation seeds,
separate from the 60000–60499 held-out benchmark family. The callback does
not train on the selected evaluation episodes or modify the shipped policy.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from .rollouts import run_episode


class SeededValidationCallback(BaseCallback):
    """Save the checkpoint with the highest held-out validation success rate."""

    def __init__(self, *, make_env, counts, eval_freq: int, episodes: int,
                 seed_start: int, destination: str):
        super().__init__(verbose=0)
        if episodes < 1 or eval_freq < 1:
            raise ValueError("Validation episodes and frequency must be positive")
        if not (seed_start + episodes - 1 < 60_000 or seed_start > 60_499):
            raise ValueError("Validation seeds overlap the published held-out evaluation range")
        self.make_env = make_env
        self.counts = counts
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.seed_start = seed_start
        self.destination = Path(destination)
        self.best_score = float("-inf")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq:
            return True
        total, success, collision, timeout = 0, 0, 0, 0
        by_count = []
        for n in self.counts:
            env = self.make_env(n)
            local = {"success": 0, "collision": 0, "timeout": 0}
            try:
                for i in range(self.episodes):
                    metrics, _ = run_episode(env, self.model, self.seed_start + i, None)
                    local[metrics.outcome] += 1
            finally:
                env.close()
            total += self.episodes
            success += local["success"]
            collision += local["collision"]
            timeout += local["timeout"]
            by_count.append({"obstacles": n, **local})
        score = success / total
        self.logger.record("validation/success_rate", score)
        self.logger.record("validation/collision_rate", collision / total)
        self.logger.record("validation/timeout_rate", timeout / total)
        self.destination.mkdir(parents=True, exist_ok=True)
        csv_path = self.destination / "validation_history.csv"
        with csv_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            if fp.tell() == 0:
                writer.writerow(["timesteps", "successes", "collisions", "timeouts", "episodes", "success_rate"])
            writer.writerow([self.num_timesteps, success, collision, timeout, total, score])
        if score > self.best_score:
            self.best_score = score
            path = self.destination / "validation_best"
            self.model.save(str(path))
            meta = {"status": "validation_not_test", "model_timesteps": self.num_timesteps,
                    "validation_seed_start": self.seed_start, "episodes_per_count": self.episodes,
                    "stage_obstacle_counts": self.counts, "success": success,
                    "collision": collision, "timeout": timeout,
                    "success_rate": score, "by_count": by_count}
            (self.destination / "validation_best.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"Validation at {self.num_timesteps} steps: "
              f"{success}/{total} success, {collision} collisions, {timeout} timeouts")
        return True
