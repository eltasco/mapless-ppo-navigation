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

    train_p = sub.add_parser("train", help="Train from scratch or fine-tune existing PPO")
    train_p.add_argument("model", help="scratch or checkpoint path (best supported)")
    train_p.add_argument("difficulty", help="mixed | hard | easy | ...")
    train_p.add_argument("obstacles", help="Contiguous range, e.g. 2-5 or 3")
    train_p.add_argument("timesteps", type=int)
    train_p.add_argument("output", help="Output model .zip path")
    train_p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    train_p.add_argument("--checkpoint-every", type=int, default=5000)
    train_p.add_argument("--seed", type=int, default=42)

    view_p = sub.add_parser("view", help="Visualize a checkpoint in Pygame")
    view_p.add_argument("model", help="Checkpoint path, or best")
    view_p.add_argument("difficulty", help="One curriculum stage")
    view_p.add_argument("obstacles", type=int, help="Number of circular obstacles")
    view_p.add_argument("radius", nargs="?", default="original")
    view_p.add_argument("--fps", type=int, default=30)

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
                summary_json=args.json,
            )
        if args.command == "train":
            parse_stages(args.difficulty)  # Validates the difficulty immediately.
            return train(
                args.model, args.difficulty, args.obstacles, args.timesteps,
                args.output, learning_rate=args.lr,
                checkpoint_every=args.checkpoint_every, seed=args.seed,
            )
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
