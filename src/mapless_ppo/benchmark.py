"""Paired PPO vs classical-controller benchmarking and failure diagnostics.

Only new evaluations populate new CSV/JSON outputs. Historical metrics are
never copied into the results of an automated benchmark.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from .baselines import PotentialFieldController
from .environment import GymNavigationEnv
from .experiments import _check_spaces, model_path, require_checkpoint
from .frame_stack import FrameStackWrapper
from .scenarios import parse_radius, set_episode_radius
from .stress import StressWorld


def wilson_interval(successes: int, total: int, z: float = 1.95996398454) -> tuple[float, float]:
    """Wilson 95% confidence interval for a binomial success probability."""
    if not total:
        raise ValueError("Total episodes must be positive")
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _csv(path: str, rows: list[dict]) -> None:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with dest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@dataclass
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


def benchmark(
    *, controllers: str = "both", checkpoint: str = "best",
    stage: str = "hard", counts: list[int] | None = None,
    episodes: int = 100, seed_start: int = 60_000, radius_spec: str = "original",
    layout: str = "original", noise_std: float = 0.0, dropout: float = 0.0,
    csv_path: str = "results/paired_benchmark.csv",
    json_path: str = "results/paired_benchmark.json",
    trace_csv: str | None = None, trace_timeouts: int = 3,
) -> list[BenchmarkSummary]:
    if controllers not in {"both", "ppo", "potential"}:
        raise ValueError("controllers must be both, ppo or potential")
    if episodes < 1 or seed_start < 0 or trace_timeouts < 0:
        raise ValueError("Invalid episode count, seed or trace limit")
    if not counts or any(n < 1 for n in counts):
        raise ValueError("At least one positive obstacle count is required")
    if layout not in StressWorld.VALID_LAYOUTS:
        raise ValueError("Invalid layout")
    # Validate sensor parameters before running an expensive evaluation.
    if not math.isfinite(noise_std) or noise_std < 0 or not math.isfinite(dropout) or not 0 <= dropout <= 1:
        raise ValueError("Noise std must be nonnegative and dropout must be in [0,1]")
    radius = parse_radius(radius_spec)
    names = ["ppo", "potential"] if controllers == "both" else [controllers]
    checkpoint_file = None
    checkpoint_sha256 = None
    model = None
    if "ppo" in names:
        from stable_baselines3 import PPO
        checkpoint_file = model_path(checkpoint)
        require_checkpoint(checkpoint_file)
        checkpoint_sha256 = hashlib.sha256(checkpoint_file.read_bytes()).hexdigest()
        model = PPO.load(str(checkpoint_file), device="cpu")

    rows: list[dict] = []
    failure_traces: list[dict] = []
    summaries: list[BenchmarkSummary] = []
    paired: list[dict] = []
    for n in counts:
        outcomes_by_controller: dict[str, dict[int, str]] = {}
        for name in names:
            controller = model if name == "ppo" else PotentialFieldController()
            env = FrameStackWrapper(StressWorld(GymNavigationEnv(n, stage), layout=layout, noise_std=noise_std, dropout=dropout))
            if name == "ppo":
                _check_spaces(model, env)
            counts_seen: Counter = Counter()
            total_steps, total_reward = 0, 0.0
            outcome_map = {}
            trace_saved = 0
            try:
                for i in range(episodes):
                    seed = seed_start + i
                    set_episode_radius(env, radius, seed)
                    obs, _ = env.reset(seed=seed)
                    initial_obstacles = tuple(env.robot.obstacles)
                    reward_sum = 0.0
                    path_length = 0.0
                    action_flips = 0
                    previous_v = None
                    small_velocity_steps = 0
                    min_clearance = math.inf
                    trace = []
                    while True:
                        action, _ = controller.predict(obs, deterministic=True)
                        old_x, old_y = env.robot.x, env.robot.y
                        obs, reward, terminated, truncated, info = env.step(action)
                        v = float(action[0])
                        w = float(action[1])
                        path_length += math.hypot(env.robot.x - old_x, env.robot.y - old_y)
                        reward_sum += float(reward)
                        if abs(v) < 0.10:
                            small_velocity_steps += 1
                        if previous_v is not None and previous_v * v < -0.04:
                            action_flips += 1
                        previous_v = v
                        min_clearance = min(min_clearance, float(info["clearance"]))
                        if trace_csv is not None and trace_saved < trace_timeouts:
                            trace.append({
                                "controller": name, "stage": stage, "obstacles": n,
                                "seed": seed, "step": env.robot.steps,
                                "x_m": round(env.robot.x, 6), "y_m": round(env.robot.y, 6),
                                "yaw_rad": round(env.robot.theta, 6),
                                "linear_action": round(v, 6), "angular_action": round(w, 6),
                                "goal_distance_m": round(env.robot.get_observation()[0], 6),
                                "clearance_m": round(float(info["clearance"]), 6),
                            })
                        if terminated or truncated:
                            break
                    outcome = "success" if info["goal_reached"] else "collision" if info["collision"] else "timeout"
                    counts_seen[outcome] += 1
                    outcome_map[seed] = outcome
                    total_steps += env.robot.steps
                    total_reward += reward_sum
                    if outcome == "timeout" and trace_csv is not None and trace_saved < trace_timeouts:
                        failure_traces.extend(trace)
                        trace_saved += 1
                    rows.append({
                        "controller": name, "stage": stage, "obstacles": n,
                        "seed": seed, "layout": layout, "radius_m": round(env.robot.obstacle_radius, 6),
                        "noise_std_m": noise_std, "dropout": dropout,
                        "outcome": outcome, "steps": env.robot.steps,
                        "reward": round(reward_sum, 6), "path_length_m": round(path_length, 6),
                        "min_clearance_m": round(min_clearance, 6),
                        "low_speed_fraction": round(small_velocity_steps / env.robot.steps, 6),
                        "velocity_sign_flips": action_flips,
                        "end_goal_distance_m": round(env.robot.get_observation()[0], 6),
                    })
                    if tuple(env.robot.obstacles) != initial_obstacles:
                        raise RuntimeError("Obstacle configuration mutated during rollout")
            finally:
                env.close()
            lo, hi = wilson_interval(counts_seen["success"], episodes)
            summaries.append(BenchmarkSummary(
                controller=name, stage=stage, obstacles=n, layout=layout, episodes=episodes,
                success=counts_seen["success"], collision=counts_seen["collision"],
                timeout=counts_seen["timeout"],
                success_rate_pct=round(100 * counts_seen["success"] / episodes, 2),
                wilson95_low_pct=round(100 * lo, 2), wilson95_high_pct=round(100 * hi, 2),
                mean_steps=round(total_steps / episodes, 2),
                mean_reward=round(total_reward / episodes, 4),
            ))
            outcomes_by_controller[name] = outcome_map
            print(f"{name:9s} | {n} {stage} / {layout} | success {counts_seen['success']}/{episodes} "
                  f"({100 * counts_seen['success']/episodes:.1f}%, Wilson 95%: {100*lo:.1f}–{100*hi:.1f}%) "
                  f"| collisions {counts_seen['collision']} | timeouts {counts_seen['timeout']}",flush=True)
        if len(names) == 2:
            ppo = outcomes_by_controller["ppo"]
            potential = outcomes_by_controller["potential"]
            paired.append({
                "obstacles": n, "episodes": episodes,
                "both_success": sum(ppo[s] == potential[s] == "success" for s in ppo),
                "ppo_only_success": sum(ppo[s] == "success" and potential[s] != "success" for s in ppo),
                "potential_only_success": sum(potential[s] == "success" and ppo[s] != "success" for s in ppo),
                "neither_success": sum(ppo[s] != "success" and potential[s] != "success" for s in ppo),
            })
    _csv(csv_path, rows)
    if trace_csv is not None:
        _csv(trace_csv, failure_traces)
    metadata = {
        "status": "newly_measured_not_historical",
        "model": str(checkpoint_file) if checkpoint_file else None,
        "model_sha256": checkpoint_sha256,
        "seed_start": seed_start, "episodes_per_configuration": episodes,
        "stage": stage, "obstacle_counts": counts, "layout": layout,
        "noise_std_m": noise_std, "dropout_probability": dropout,
        "radius_spec": radius_spec, "controller_comparison": asdict(PotentialFieldController()),
        "summaries": [asdict(s) for s in summaries], "paired_outcomes": paired,
        "limitations": "Idealized kinematics and static circular obstacles, even under layout/noise stress tests."
    }
    output_json = Path(json_path)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {csv_path}, {json_path}" + (f", {trace_csv}" if trace_csv else ""))
    return summaries
