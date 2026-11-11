"""PPO training, checkpointing, deterministic evaluation, and machine-readable results."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from .environment import GymNavigationEnv, STAGES
from .scenarios import VariableObstacleCount, parse_radius, set_episode_radius
from .frame_stack import FrameStackWrapper
from .observation import FixedObservationScale
from .rollouts import run_episode

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


def check_checkpoint_observations(model_file: str | Path, history: str, observation_scale: str):
    """Reject known incompatible preprocessing even if feature shape matches.

    We have explicit provenance for the published policy (raw 57D) and for
    newly created policies with JSON sidecars. Older third-party checkpoints
    without sidecars are allowed with unscaled observations only.
    """
    path = actual_zip(model_file)
    shipped = actual_zip(REFERENCE_MODEL)
    if path.resolve() == shipped.resolve():
        if (history, observation_scale) != ("stack", "none"):
            raise ValueError("The released model requires --history stack and --observation-scale none")
        return
    metadata = path.with_suffix(".json")
    if metadata.exists():
        info = json.loads(metadata.read_text(encoding="utf-8"))
        if info.get("history", "stack") != history or info.get("observation_scale", "none") != observation_scale:
            raise ValueError("Checkpoint's saved preprocessing differs from requested options")
    elif observation_scale != "none":
        raise ValueError("Fixed scaling requires a checkpoint with a JSON preprocessing sidecar")


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


def environment(count: int, stage: str, *, history: str = "stack",
                risk_weight: float = 1.5, observation_scale: str = "none"):
    """Build an env matching the chosen observation contract.

    Default `none` preserves the bundled 57-D checkpoint unchanged. Fixed
    scaling is opt-in and requires separately trained weights.
    """
    base = GymNavigationEnv(count, stage, risk_weight=risk_weight)
    if observation_scale == "fixed":
        base = FixedObservationScale(base)
    elif observation_scale != "none":
        raise ValueError("observation_scale must be 'none' or 'fixed'")
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
    radius_spec: str = "original", episodes: int = 500,
    seed_start: int = 60_000, progress_every: int = 50,
    episode_csv: str | None = None, summary_json: str | None = None,
    history: str = "stack", observation_scale: str = "none",
    layout: str = "original", noise_std: float = 0.0, dropout: float = 0.0,
) -> list[EvaluationSummary]:
    """Evaluate one checkpoint using the exact shared rollout implementation.

    Supports 19-D/57-D policies and the same OOD settings as benchmark.
    """
    from .benchmark import make_benchmark_env
    from .stress import StressWorld
    if episodes < 1 or progress_every < 1 or seed_start < 0:
        raise ValueError("Invalid evaluation arguments")
    if layout not in StressWorld.VALID_LAYOUTS:
        raise ValueError("Invalid layout")
    if noise_std < 0 or not math.isfinite(noise_std) or not 0 <= dropout <= 1:
        raise ValueError("Invalid LiDAR noise or dropout")
    model_file = model_path(checkpoint)
    require_checkpoint(model_file)
    check_checkpoint_observations(model_file, history, observation_scale)
    model = PPO.load(str(model_file), device="cpu")
    radius = parse_radius(radius_spec)
    summaries: list[EvaluationSummary] = []
    episode_rows = []
    for stage in stages:
        for count in counts:
            env = make_benchmark_env(count, stage, layout, noise_std, dropout,
                                     history, observation_scale)
            _check_spaces(model, env)
            outcomes = {"success": 0, "collision": 0, "timeout": 0}
            total_reward = total_steps = 0
            try:
                for i in range(episodes):
                    seed = seed_start + i
                    metrics, _ = run_episode(env, model, seed, radius)
                    outcomes[metrics.outcome] += 1
                    total_reward += metrics.reward
                    total_steps += metrics.steps
                    episode_rows.append({
                        "model": str(model_file), "stage": stage, "obstacles": count,
                        "seed": seed, "layout": layout, "history": history,
                        "observation_scale": observation_scale,
                        "noise_std_m": noise_std, "dropout": dropout,
                        "radius_m": round(metrics.radius_m, 6),
                        "outcome": metrics.outcome, "steps": metrics.steps,
                        "reward": round(metrics.reward, 6),
                        "low_speed_fraction": round(metrics.low_speed_fraction, 6),
                        "velocity_sign_flips": metrics.velocity_sign_flips,
                    })
                    if (i + 1) % progress_every == 0 or i + 1 == episodes:
                        print(f"[{count} obstacles / {stage} / {layout}] "
                              f"{i+1}/{episodes} | success {outcomes['success']/(i+1):.1%}",
                              flush=True)
            finally:
                env.close()
            summary = EvaluationSummary(
                model=str(model_file), stage=stage, obstacles=count, radius=radius_spec,
                episodes=episodes, seed_start=seed_start,
                success=outcomes["success"], collision=outcomes["collision"],
                timeout=outcomes["timeout"], success_rate=100*outcomes["success"]/episodes,
                mean_reward=total_reward/episodes, mean_steps=total_steps/episodes,
            )
            summaries.append(summary)
            print(f"{count} obstacles / {stage} / {layout}: success={summary.success_rate:.1f}%, "
                  f"collisions={summary.collision}, timeouts={summary.timeout}")
    if episode_csv and episode_rows:
        from .benchmark import write_csv
        write_csv(episode_csv, episode_rows)
    if summary_json:
        p = Path(summary_json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "status": "newly_measured", "layout": layout, "noise_std_m": noise_std,
            "dropout_probability": dropout, "history": history,
            "observation_scale": observation_scale,
            "evaluation": [asdict(s) for s in summaries],
        }, indent=2) + "\n", encoding="utf-8")
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
    validation_episodes: int = 50, validation_seed_start: int = 20_000,
    observation_scale: str = "none",
) -> str:
    if (history not in {"stack", "single"} or observation_scale not in {"none", "fixed"}
        or not math.isfinite(risk_weight) or risk_weight < 0 or eval_every < 0):
        raise ValueError("Invalid training history, risk weight, or eval frequency")
    if validation_episodes < 1 or validation_seed_start < 0:
        raise ValueError("Invalid validation episode count or seed")
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
            environment(counts[0], stage, history=history, risk_weight=risk_weight,
                        observation_scale=observation_scale), counts[0], counts[-1]
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
            check_checkpoint_observations(start_path, history, observation_scale)
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
        if eval_every > 0:
            from .validation import SeededValidationCallback
            callbacks.append(SeededValidationCallback(
                make_env=lambda n: environment(n, stage, history=history,
                                                risk_weight=risk_weight,
                                                observation_scale=observation_scale),
                counts=counts, eval_freq=eval_every,
                episodes=validation_episodes, seed_start=validation_seed_start,
                destination=str(checkpoint_dir / "validation"),
            ))
        print(
            f"Training from {start_label} | stage={stage} | obstacles={counts_spec} "
            f"| steps={timesteps} | lr={model.learning_rate:g}"
        )
        model.learn(
            total_timesteps=timesteps,
            reset_num_timesteps=start_label == "scratch",
            callback=CallbackList(callbacks),
        )
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
            "observation_scale": observation_scale,
            "validation_episodes_per_count": validation_episodes,
            "validation_seed_start": validation_seed_start,
            "eval_every": eval_every, "tensorboard_log": tensorboard_log,
            "note": "On-policy PPO may process a full final rollout beyond requested timesteps.",
        }
        out.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved model: {out}\nMetadata: {out.with_suffix('.json')}")
        return str(out)
    finally:
        env.close()
