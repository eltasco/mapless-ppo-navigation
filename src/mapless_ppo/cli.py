"""Unified command-line interface, with evaluation, visualization and training commands."""

from __future__ import annotations

import argparse
from .experiments import evaluate, model_path, parse_counts, parse_stages, train
from .scenarios import parse_radius


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mapless-ppo",
        description="Mapless goal navigation with PPO and LiDAR frame history",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    eval_p = sub.add_parser("eval", help="Run deterministic seeded evaluation")
    eval_p.add_argument("model", help="Checkpoint path, or best")
    eval_p.add_argument("difficulty", help="easy | medium | medium_hard | hard | hard_passable | mixed | all")
    eval_p.add_argument("obstacles", help="5 | 2-5 | 2,4,5 | all")
    eval_p.add_argument("radius", nargs="?", default="original", help="original | 0.50 | 0.25-0.50")
    eval_p.add_argument("--episodes", type=int, default=500)
    eval_p.add_argument("--seed-start", type=int, default=60000)
    eval_p.add_argument("--progress-every", type=int, default=50)
    eval_p.add_argument("--csv", default=None, help="Write one row per episode")
    eval_p.add_argument("--json", default=None, help="Save evaluation summary as JSON")
    eval_p.add_argument("--layout", choices=["original", "uniform", "long_goal"], default="original")
    eval_p.add_argument("--lidar-noise", type=float, default=0.0)
    eval_p.add_argument("--dropout", type=float, default=0.0)
    eval_p.add_argument("--observation-scale", choices=["none", "fixed"], default="none")
    eval_p.add_argument("--history", choices=["stack", "single"], default="stack",
                        help="Observation format: stacked 57-D or single 19-D")

    train_p = sub.add_parser("train", help="Train from scratch or fine-tune existing PPO")
    train_p.add_argument("model", help="scratch or checkpoint path (best supported)")
    train_p.add_argument("difficulty", help="mixed | hard | easy | ...")
    train_p.add_argument("obstacles", help="Contiguous range, e.g. 2-5 or 3")
    train_p.add_argument("timesteps", type=int)
    train_p.add_argument("output", help="Output model .zip path")
    train_p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_p.add_argument("--checkpoint-every", type=int, default=5000)
    train_p.add_argument("--seed", type=int, default=42)
    train_p.add_argument("--history", choices=["stack", "single"], default="stack")
    train_p.add_argument("--risk-weight", type=float, default=1.5,
                         help="Closing-speed penalty coefficient; 0 removes the risk term")
    train_p.add_argument("--eval-every", type=int, default=0,
                         help="Validation frequency (0 disables periodic evaluation)")
    train_p.add_argument("--validation-episodes", type=int, default=50,
                         help="Episodes per obstacle count during validation (default 50)")
    train_p.add_argument("--validation-seed-start", type=int, default=20_000)
    train_p.add_argument("--observation-scale", choices=["none", "fixed"], default="none")
    train_p.add_argument("--tensorboard-log", default=None,
                         help="TensorBoard log directory (requires tensorboard)")

    view_p = sub.add_parser("view", help="Visualize a checkpoint in Pygame")
    view_p.add_argument("model", help="Checkpoint path, or best")
    view_p.add_argument("difficulty", help="One curriculum stage")
    view_p.add_argument("obstacles", type=int, help="Number of circular obstacles")
    view_p.add_argument("radius", nargs="?", default="original")
    view_p.add_argument("--fps", type=int, default=30)

    benchmark_p = sub.add_parser("benchmark", help="Compare PPO and a potential-field baseline on paired scenes")
    benchmark_p.add_argument("--controllers", choices=["both", "both_legacy", "ppo", "potential", "potential_legacy"], default="both")
    benchmark_p.add_argument("--baseline-config", default=None,
                             help="JSON file from tune-baseline; otherwise gains are NOT calibrated")
    benchmark_p.add_argument("--history", choices=["stack", "single"], default="stack")
    benchmark_p.add_argument("--observation-scale", choices=["none", "fixed"], default="none")
    benchmark_p.add_argument("--model", default="best", help="Reference PPO checkpoint")
    benchmark_p.add_argument("--difficulty", default="hard", help="One curriculum difficulty")
    benchmark_p.add_argument("--obstacles", default="2-5", help="Obstacle counts, e.g. 2-5")
    benchmark_p.add_argument("--episodes", type=int, default=100)
    benchmark_p.add_argument("--seed-start", type=int, default=60_000)
    benchmark_p.add_argument("--radius", default="original")
    benchmark_p.add_argument("--layout", choices=["original", "uniform", "long_goal"], default="original")
    benchmark_p.add_argument("--lidar-noise", type=float, default=0.0, help="Gaussian LiDAR noise sigma in meters")
    benchmark_p.add_argument("--dropout", type=float, default=0.0, help="Probability a LiDAR ray becomes max range")
    benchmark_p.add_argument("--csv", default="results/paired_benchmark.csv")
    benchmark_p.add_argument("--json", default="results/paired_benchmark.json")
    benchmark_p.add_argument("--trace-csv", default=None, help="Write selected timeout trajectories")
    benchmark_p.add_argument("--trace-timeouts", type=int, default=3)

    tune_p = sub.add_parser("tune-baseline", help="Tune classical controller on disjoint validation seeds")
    tune_p.add_argument("--difficulty", default="hard")
    tune_p.add_argument("--obstacles", default="2-5")
    tune_p.add_argument("--episodes", type=int, default=30,
                        help="Validation episodes per obstacle count and candidate")
    tune_p.add_argument("--seed-start", type=int, default=30_000)
    tune_p.add_argument("--output", default="configs/potential_validated.json")
    tune_p.add_argument("--force", action="store_true")

    return p


def main(argv: list[str] | None = None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "eval":
            return evaluate(
                args.model, parse_stages(args.difficulty), parse_counts(args.obstacles),
                radius_spec=args.radius, episodes=args.episodes, seed_start=args.seed_start,
                progress_every=args.progress_every, episode_csv=args.csv,
                summary_json=args.json, history=args.history,
                observation_scale=args.observation_scale, layout=args.layout,
                noise_std=args.lidar_noise, dropout=args.dropout,
            )
        if args.command == "train":
            parse_stages(args.difficulty)  # Validates the difficulty immediately.
            return train(
                args.model, args.difficulty, args.obstacles, args.timesteps,
                args.output, learning_rate=args.lr,
                checkpoint_every=args.checkpoint_every, seed=args.seed,
                history=args.history, risk_weight=args.risk_weight,
                eval_every=args.eval_every, tensorboard_log=args.tensorboard_log,
                validation_episodes=args.validation_episodes,
                validation_seed_start=args.validation_seed_start,
                observation_scale=args.observation_scale,
            )
        if args.command == "benchmark":
            stages = parse_stages(args.difficulty)
            if len(stages) != 1:
                raise ValueError("Benchmark requires one difficulty, not all")
            from .benchmark import benchmark
            return benchmark(controllers=args.controllers, checkpoint=args.model,
                             stage=stages[0], counts=parse_counts(args.obstacles),
                             episodes=args.episodes, seed_start=args.seed_start,
                             radius_spec=args.radius, layout=args.layout,
                             noise_std=args.lidar_noise, dropout=args.dropout,
                             csv_path=args.csv, json_path=args.json,
                             trace_csv=args.trace_csv, trace_timeouts=args.trace_timeouts,
                             history=args.history, observation_scale=args.observation_scale,
                             baseline_config=args.baseline_config)
        if args.command == "tune-baseline":
            from .tuning import tune_baseline
            stages = parse_stages(args.difficulty)
            if len(stages) != 1:
                raise ValueError("Tuning requires exactly one difficulty")
            return tune_baseline(stage=stages[0], counts=parse_counts(args.obstacles),
                                 seed_start=args.seed_start, episodes=args.episodes,
                                 output=args.output, force=args.force)
        if args.command == "view":
            if args.obstacles < 1 or args.fps < 1:
                raise ValueError("Obstacles and FPS must be positive")
            stages = parse_stages(args.difficulty)
            if len(stages) != 1:
                raise ValueError("Viewer requires one difficulty, not 'all'")
            parse_radius(args.radius)
            from .visualization import view
            return view(
                model_path(args.model), stages[0], args.obstacles,
                radius_spec=args.radius, fps=args.fps,
            )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
