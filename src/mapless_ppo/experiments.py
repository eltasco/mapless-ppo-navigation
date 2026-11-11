"""PPO training, checkpointing, deterministic evaluation, and machine-readable results."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from .environment import GymNavigationEnv, STAGES
from .scenarios import VariableObstacleCount, parse_radius, set_episode_radius
from .frame_stack import FrameStackWrapper

REFERENCE_MODEL = "models/navigation_ppo.zip"


def model_path(path: str) -> Path:
    """Resolve common aliases, preserving explicit and nested checkpoint paths."""
    key = path.lower().strip()
    if key in ("reference", "best"):
        return Path(REFERENCE_MODEL)
    value = Path(path)
    if value.exists() or value.with_suffix(".zip").exists():
        return value
    candidate = Path("models") / value
    return candidate if candidate.exists() or candidate.with_suffix(".zip").exists() else value


def actual_zip(path: str | Path) -> Path:
    p = Path(path)
    return p if p.suffix == ".zip" else Path(f"{p}.zip")


def require_checkpoint(path: str | Path) -> None:
    if not actual_zip(path).is_file():
        raise FileNotFoundError(
            f"Missing model: {actual_zip(path)}. "
            "Provide a Stable-Baselines3 PPO checkpoint (.zip) or train a new policy."
        )


def parse_counts(spec: str) -> list[int]:
    """Accept '3', '2-5', '2,4,5', or 'all' (=2-5)."""
    spec = str(spec).lower().strip()
    if spec in {"all", "*"}:
        spec = "2-5"
    counts: set[int] = set()
    for chunk in spec.split(","):
        if "-" in chunk:
            parts = chunk.split("-")
            if len(parts) != 2:
                raise ValueError("Invalid obstacle range")
            lo, hi = (int(x) for x in parts)
            if lo < 1 or hi < lo or hi - lo > 99:
                raise ValueError("Invalid obstacle range")
            counts.update(range(lo, hi + 1))
        else:
            counts.add(int(chunk))
    if not counts or min(counts) < 1 or len(counts) > 100:
        raise ValueError("Obstacle count must be positive; maximum 100 different counts")
    return sorted(counts)


def parse_stages(spec: str) -> list[str]:
    spec = spec.lower().replace("-", "_")
    if spec in {"all", "*"}:
        return list(STAGES)
    if spec not in STAGES:
        raise ValueError(f"Unknown difficulty {spec!r}. Choose from {STAGES} or 'all'.")
    return [spec]


def environment(count: int, stage: str, *, history: str = "stack", risk_weight: float = 1.5):
    """Create a Gymnasium environment; default matches the shipped 57-D policy."""
    base = GymNavigationEnv(count, stage, risk_weight=risk_weight)
    if history == "stack":
        return FrameStackWrapper(base)
    if history == "single":
        return base
    raise ValueError("history must be 'stack' or 'single'")


def _check_spaces(model: PPO, env) -> None:
    if model.observation_space.shape != env.observation_space.shape or model.action_space.shape != env.action_space.shape:
        raise ValueError(
            f"Model space {model.observation_space.shape}/{model.action_space.shape}, "
            f"environment space {env.observation_space.shape}/{env.action_space.shape}: incompatible."
        )


@dataclass(frozen=True)
class EvaluationSummary:
    model: str
    stage: str
    obstacles: int
    radius: str
    episodes: int
    seed_start: int
    success: int
    collision: int
    timeout: int
    success_rate: float
    mean_reward: float
    mean_steps: float


def evaluate(
    checkpoint: str,
    stages: list[str],
    counts: list[int],
    *,
    radius_spec: str = "original",
    episodes: int = 500,
    seed_start: int = 60_000,
    progress_every: int = 50,
    episode_csv: str | None = None,
    summary_json: str | None = None,
    history: str = "stack",
) -> list[EvaluationSummary]:
    if episodes < 1 or progress_every < 1:
        raise ValueError("Episodes and progress interval must be positive")
    model_file = model_path(checkpoint)
    require_checkpoint(model_file)
    model = PPO.load(str(model_file), device="cpu")
    radius = parse_radius(radius_spec)
    summaries: list[EvaluationSummary] = []
    episode_rows = []

    for stage in stages:
        for count in counts:
            env = environment(count, stage, history=history)
            _check_spaces(model, env)
            outcomes = {"success": 0, "collision": 0, "timeout": 0}
            total_reward = 0.0
            total_steps = 0
            try:
                for ep in range(episodes):
                    seed = seed_start + ep
                    episode_radius = set_episode_radius(env, radius, seed)
                    obs, _ = env.reset(seed=seed)
                    if not math.isclose(env.robot.obstacle_radius, episode_radius, abs_tol=1e-9):
                        raise RuntimeError("Environment unexpectedly changed obstacle radius during reset")
                    if len(env.robot.obstacles) != count:
                        raise RuntimeError("Obstacle count mismatch")
                    if not env.observation_space.contains(obs):
                        raise RuntimeError("Invalid initial observation")
                    episode_reward = 0.0
                    done = False
                    while not done:
                        action, _ = model.predict(obs, deterministic=True)
                        obs, reward, terminated, truncated, info = env.step(action)
                        episode_reward += float(reward)
                        done = bool(terminated or truncated)
                    outcome = (
                        "success" if info.get("goal_reached", False)
                        else "collision" if info.get("collision", False)
                        else "timeout"
                    )
                    outcomes[outcome] += 1
                    total_reward += episode_reward
                    total_steps += env.robot.steps
                    episode_rows.append({
                        "model": str(model_file), "stage": stage, "obstacles": count,
                        "seed": seed, "radius_m": round(episode_radius, 6),
                        "outcome": outcome, "steps": env.robot.steps,
                        "reward": round(episode_reward, 6),
                    })
                    if (ep + 1) % progress_every == 0 or ep + 1 == episodes:
                        print(
                            f"[{count} obstacles / {stage}] {ep + 1}/{episodes} "
                            f"| success {outcomes['success'] / (ep + 1):.1%}",
                            flush=True,
                        )
            finally:
                env.close()
            summary = EvaluationSummary(
                model=str(model_file), stage=stage, obstacles=count, radius=radius_spec,
                episodes=episodes, seed_start=seed_start,
                success=outcomes["success"], collision=outcomes["collision"],
                timeout=outcomes["timeout"],
                success_rate=100.0 * outcomes["success"] / episodes,
                mean_reward=total_reward / episodes,
                mean_steps=total_steps / episodes,
            )
            summaries.append(summary)
            print(
                f"\n{count} OBSTACLES / {stage.upper()} | {summary.success_rate:.1f}% "
                f"({summary.success}/{episodes}) | collisions={summary.collision} "
                f"| timeouts={summary.timeout} | mean reward={summary.mean_reward:.2f}\n"
            )

    if episode_csv:
        p = Path(episode_csv)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0]))
            writer.writeheader()
            writer.writerows(episode_rows)
        print(f"Episode results: {p}")
    if summary_json:
        p = Path(summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps({
                "generated_utc": datetime.now(timezone.utc).isoformat(),
                "evaluation": [asdict(s) for s in summaries],
            }, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Summary: {p}")

    print("SUMMARY")
    for s in summaries:
        print(f"  {s.stage:12s} | {s.obstacles} obstacles | {s.success_rate:5.1f}%")
    return summaries


def train(
    start: str,
    stage: str,
    counts_spec: str,
    timesteps: int,
    output: str,
    *,
    learning_rate: float | None = None,
    checkpoint_every: int = 5_000,
    seed: int = 42,
    history: str = "stack",
    risk_weight: float = 1.5,
    eval_every: int = 0,
    tensorboard_log: str | None = None,
) -> str:
    if history not in {"stack", "single"} or not math.isfinite(risk_weight) or risk_weight < 0 or eval_every < 0:
        raise ValueError("Invalid training history, risk weight, or eval frequency")
    if timesteps < 1 or checkpoint_every < 1:
        raise ValueError("Timesteps and checkpoint frequency must be positive")
    counts = parse_counts(counts_spec)
    if counts != list(range(counts[0], counts[-1] + 1)):
        raise ValueError("Training requires a contiguous obstacle interval (e.g. 2-5)")
    if stage not in STAGES:
        raise ValueError(f"Invalid training difficulty: {stage}")
    out = actual_zip(output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite an existing model: {out}")
    if learning_rate is not None and learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    # The ordering matters: count chosen prior to FrameStackWrapper.reset().
    env = Monitor(
        VariableObstacleCount(
            environment(counts[0], stage, history=history, risk_weight=risk_weight), counts[0], counts[-1]
        ),
        info_keywords=("num_obstacles",),
    )
    try:
        if start.lower() in {"scratch", "new"}:
            lr = 3e-4 if learning_rate is None else learning_rate
            model = PPO(
                "MlpPolicy", env, learning_rate=lr, n_steps=2048,
                batch_size=64, n_epochs=10, gamma=0.995,
                gae_lambda=0.95, clip_range=0.2,
                policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64])),
                seed=seed, device="cpu", verbose=1, tensorboard_log=tensorboard_log,
            )
            start_label = "scratch"
        else:
            start_path = model_path(start)
            require_checkpoint(start_path)
            if actual_zip(start_path).resolve() == out.resolve():
                raise ValueError("Input and output checkpoint paths must differ")
            # SB3.load accepts an overridden learning rate while retaining
            # the learned weights; no reinitialization of the policy is done.
            kwargs = {"learning_rate": learning_rate} if learning_rate is not None else {}
            if tensorboard_log is not None:
                kwargs["tensorboard_log"] = tensorboard_log
            model = PPO.load(str(start_path), env=env, device="cpu", **kwargs)
            start_label = str(start_path)
        _check_spaces(model, env)
        out.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path("models/checkpoints") / out.stem
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        callbacks = [CheckpointCallback(
            save_freq=checkpoint_every,
            save_path=str(checkpoint_dir),
            name_prefix="ppo_navigation",
        )]
        eval_env = None
        if eval_every > 0:
            # Validation is kept separate from held-out seed ranges used below.
            eval_env = Monitor(environment(counts[-1], stage, history=history, risk_weight=risk_weight))
            eval_env.reset(seed=seed + 10_000)
            callbacks.append(EvalCallback(
                eval_env, best_model_save_path=str(checkpoint_dir / "validation_best"),
                log_path=str(checkpoint_dir / "validation"),
                eval_freq=eval_every, n_eval_episodes=10,
                deterministic=True, verbose=1,
            ))
        print(
            f"Training from {start_label} | stage={stage} | obstacles={counts_spec} "
            f"| steps={timesteps} | lr={model.learning_rate:g}"
        )
        try:
            model.learn(
                total_timesteps=timesteps,
                reset_num_timesteps=start_label == "scratch",
                callback=CallbackList(callbacks),
            )
        finally:
            if eval_env is not None:
                eval_env.close()
        model.save(str(out))
        metadata = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(out), "started_from": start_label,
            "difficulty": stage, "obstacle_counts": counts,
            "requested_timesteps": timesteps,
            "actual_cumulative_timesteps": int(model.num_timesteps),
            "learning_rate": float(model.learning_rate),
            "seed": seed, "observations": list(env.observation_space.shape),
            "history": history, "risk_weight": risk_weight,
            "eval_every": eval_every, "tensorboard_log": tensorboard_log,
            "note": "On-policy PPO may process a full final rollout beyond requested timesteps.",
        }
        out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved model: {out}\nMetadata: {out.with_suffix('.json')}")
        return str(out)
    finally:
        env.close()
