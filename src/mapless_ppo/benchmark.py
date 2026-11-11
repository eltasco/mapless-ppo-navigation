"""Paired simulation benchmarks, including explicit baseline provenance.

The historical untuned Potential Field and the redesigned configurable field
are intentionally separate controllers. A calibrated baseline is marked tuned
only when its JSON config was produced by the validation-only tuning command.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from .baselines import LegacyPotentialFieldController, PotentialFieldController, load_baseline_config
from .environment import GymNavigationEnv
from .experiments import _check_spaces, model_path, require_checkpoint, check_checkpoint_observations
from .frame_stack import FrameStackWrapper
from .observation import FixedObservationScale
from .rollouts import run_episode
from .scenarios import parse_radius
from .stress import StressWorld


def wilson_interval(successes: int, total: int, z: float = 1.95996398454) -> tuple[float, float]:
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Invalid binomial count")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def write_csv(path: str | None, rows: list[dict]):
    if not path or not rows:
        return
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_benchmark_env(n: int, stage: str, layout: str, noise_std: float, dropout: float,
                       history: str = "stack", observation_scale: str = "none"):
    env = StressWorld(GymNavigationEnv(n, stage), layout=layout,
                      noise_std=noise_std, dropout=dropout)
    if observation_scale == "fixed":
        env = FixedObservationScale(env)
    elif observation_scale != "none":
        raise ValueError("observation_scale must be none or fixed")
    if history == "stack":
        env = FrameStackWrapper(env)
    elif history != "single":
        raise ValueError("history must be stack or single")
    return env


class _BaselineInputAdapter:
    """Undo fixed scaling for analytic baselines when comparing scaled PPO."""
    def __init__(self, baseline):
        self.baseline = baseline
        self.scale = np.array([5.0, 1.0, 1.0] + [3.0] * 16, dtype=np.float32)

    def predict(self, obs, deterministic=True):
        raw = np.asarray(obs, dtype=np.float32).reshape(-1, 19) * self.scale
        return self.baseline.predict(raw.flatten(), deterministic=deterministic)


@dataclass(frozen=True)
class BenchmarkSummary:
    controller: str
    stage: str
    obstacles: int
    layout: str
    episodes: int
    success: int
    collision: int
    timeout: int
    success_rate_pct: float
    wilson95_low_pct: float
    wilson95_high_pct: float
    mean_steps: float
    mean_reward: float


def benchmark(*, controllers: str = "both", checkpoint: str = "best", stage: str = "hard",
              counts: list[int] | None = None, episodes: int = 100, seed_start: int = 60_000,
              radius_spec: str = "original", layout: str = "original", noise_std: float = 0.0,
              dropout: float = 0.0, csv_path: str | None = "results/paired_benchmark.csv",
              json_path: str | None = "results/paired_benchmark.json",
              trace_csv: str | None = None, trace_timeouts: int = 3,
              history: str = "stack", observation_scale: str = "none",
              baseline_config: str | None = None) -> list[BenchmarkSummary]:
    if controllers not in {"both", "both_legacy", "ppo", "potential", "potential_legacy"}:
        raise ValueError("controllers must be both, both_legacy, ppo, potential or potential_legacy")
    if episodes < 1 or seed_start < 0 or trace_timeouts < 0 or not counts or any(x < 1 for x in counts):
        raise ValueError("Invalid episodes, seed, trace limit or obstacle count")
    if layout not in StressWorld.VALID_LAYOUTS or history not in ("stack", "single"):
        raise ValueError("Invalid layout or history")
    if observation_scale not in ("none", "fixed"):
        raise ValueError("Invalid observation scale")
    if not math.isfinite(noise_std) or noise_std < 0 or not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("Noise must be nonnegative and dropout in [0,1]")
    radius = parse_radius(radius_spec)
    names = (["ppo", "potential"] if controllers == "both" else
             ["ppo", "potential_legacy"] if controllers == "both_legacy" else
             [controllers])
    calibrated = load_baseline_config(baseline_config) if baseline_config else PotentialFieldController()
    baseline_name = "potential_tuned" if baseline_config else "potential_v2_untuned"
    if "potential" in names and baseline_config is None:
        print("NOTE: Potential-field v2 uses illustrative default gains, NOT calibrated gains."
              " Run `python run.py tune-baseline` on separate validation seeds.", flush=True)

    model = None
    checkpoint_file = None
    checkpoint_hash = None
    if "ppo" in names:
        from stable_baselines3 import PPO
        checkpoint_file = model_path(checkpoint)
        require_checkpoint(checkpoint_file)
        check_checkpoint_observations(checkpoint_file, history, observation_scale)
        checkpoint_file = Path(str(checkpoint_file) if Path(checkpoint_file).suffix == ".zip" else str(checkpoint_file) + ".zip")
        checkpoint_hash = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
        model = PPO.load(str(checkpoint_file), device="cpu")

    rows, traces, summaries, paired = [], [], [], []
    for n in counts:
        outcomes_by_controller = {}
        for label in names:
            display_name = (baseline_name if label == "potential" else label)
            controller = (model if label == "ppo" else
                          (LegacyPotentialFieldController() if label == "potential_legacy" else calibrated))
            if label != "ppo" and observation_scale == "fixed":
                controller = _BaselineInputAdapter(controller)
            env = make_benchmark_env(n, stage, layout, noise_std, dropout, history, observation_scale)
            if label == "ppo":
                _check_spaces(model, env)
            counts_seen = Counter()
            total_steps = 0
            total_reward = 0.0
            outcome_map = {}
            trace_saved = 0
            try:
                for i in range(episodes):
                    seed = seed_start + i
                    # Trace only the first N timeouts, not all 500 successful episodes.
                    metrics, episode_trace = run_episode(
                        env, controller, seed, radius,
                        trace=trace_csv is not None and trace_saved < trace_timeouts,
                    )
                    counts_seen[metrics.outcome] += 1
                    outcome_map[seed] = metrics.outcome
                    total_steps += metrics.steps
                    total_reward += metrics.reward
                    if metrics.outcome == "timeout" and episode_trace:
                        traces.extend([{"controller": display_name, "stage": stage,
                                        "obstacles": n, **r} for r in episode_trace])
                        trace_saved += 1
                    rows.append({
                        "controller": display_name, "stage": stage, "obstacles": n,
                        "seed": seed, "layout": layout, "radius_m": round(metrics.radius_m, 6),
                        "noise_std_m": noise_std, "dropout": dropout,
                        "history": history, "observation_scale": observation_scale,
                        "outcome": metrics.outcome, "steps": metrics.steps,
                        "reward": round(metrics.reward, 6),
                        "path_length_m": round(metrics.path_length_m, 6),
                        "min_clearance_m": round(metrics.min_clearance_m, 6),
                        "low_speed_fraction": round(metrics.low_speed_fraction, 6),
                        "velocity_sign_flips": metrics.velocity_sign_flips,
                        "end_goal_distance_m": round(metrics.end_goal_distance_m, 6),
                    })
            finally:
                env.close()
            lo, hi = wilson_interval(counts_seen["success"], episodes)
            summaries.append(BenchmarkSummary(
                controller=display_name, stage=stage, obstacles=n, layout=layout, episodes=episodes,
                success=counts_seen["success"], collision=counts_seen["collision"],
                timeout=counts_seen["timeout"],
                success_rate_pct=round(100 * counts_seen["success"] / episodes, 2),
                wilson95_low_pct=round(100 * lo, 2), wilson95_high_pct=round(100 * hi, 2),
                mean_steps=round(total_steps / episodes, 2),
                mean_reward=round(total_reward / episodes, 4),
            ))
            outcomes_by_controller[display_name] = outcome_map
            print(f"{display_name:20s} | {n} {stage} / {layout} | "
                  f"success {counts_seen['success']}/{episodes} "
                  f"({100*counts_seen['success']/episodes:.1f}%, Wilson 95%: "
                  f"{100*lo:.1f}–{100*hi:.1f}%) | collisions {counts_seen['collision']} "
                  f"| timeouts {counts_seen['timeout']}", flush=True)
        if len(outcomes_by_controller) == 2:
            a, b = list(outcomes_by_controller.values())
            paired.append({
                "obstacles": n, "episodes": episodes,
                "both_success": sum(a[s] == b[s] == "success" for s in a),
                "ppo_only_success": sum(a[s] == "success" and b[s] != "success" for s in a),
                "baseline_only_success": sum(a[s] != "success" and b[s] == "success" for s in a),
                "neither_success": sum(a[s] != "success" and b[s] != "success" for s in a),
            })
    write_csv(csv_path, rows)
    write_csv(trace_csv, traces)
    metadata = {
        "status": "measured_by_benchmark_command", "model": str(checkpoint_file) if checkpoint_file else None,
        "model_sha256": checkpoint_hash, "seed_start": seed_start,
        "episodes_per_configuration": episodes, "stage": stage, "obstacle_counts": counts,
        "layout": layout, "noise_std_m": noise_std, "dropout_probability": dropout,
        "radius_spec": radius_spec, "history": history, "observation_scale": observation_scale,
        "baseline_kind": baseline_name if "potential" in names else ("legacy" if "potential_legacy" in names else None),
        "baseline_parameters": asdict(calibrated) if "potential" in names else None,
        "baseline_config": baseline_config,
        "baseline_config_sha256": hashlib.sha256(Path(baseline_config).read_bytes()).hexdigest() if baseline_config else None,
        "summaries": [asdict(s) for s in summaries], "paired_outcomes": paired,
        "limitations": "Idealized kinematics, static circular obstacles and no global planning."
    }
    if json_path:
        out = Path(json_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("Saved " + ", ".join(str(p) for p in (csv_path, json_path, trace_csv) if p))
    return summaries
