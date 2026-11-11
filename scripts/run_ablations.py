"""Generate or execute a balanced three-seed PPO ablation matrix.

Dry-run by default. To train all nine policies, use --execute explicitly.
No checkpoint produced here is claimed to reproduce historical staged PPO.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def commands(seeds=(42, 43, 44), timesteps=200000):
    conditions = (
        ("full", "stack", "1.5"),
        ("no_history", "single", "1.5"),
        ("no_risk", "stack", "0"),
    )
    for condition, history, risk in conditions:
        for seed in seeds:
            model = f"models/ablation_{condition}_seed{seed}"
            train = [sys.executable, "run.py", "train", "scratch", "mixed", "2-5",
                     str(timesteps), model, "--seed", str(seed), "--history", history,
                     "--risk-weight", risk, "--eval-every", "10000",
                     "--validation-episodes", "50", "--validation-seed-start", "20000",
                     "--tensorboard-log", "runs"]
            evaluate = [sys.executable, "run.py", "benchmark", "--controllers", "ppo",
                        "--model", model + ".zip", "--history", history,
                        "--difficulty", "hard", "--obstacles", "2-5",
                        "--episodes", "500", "--seed-start", "60000",
                        "--csv", f"results/{condition}_seed{seed}.csv",
                        "--json", f"results/{condition}_seed{seed}.json"]
            yield train, evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run jobs (default: only print)")
    parser.add_argument("--timesteps", type=int, default=200000)
    args = parser.parse_args()
    if args.timesteps < 1:
        parser.error("Timesteps must be positive")
    for train, eval_cmd in commands(timesteps=args.timesteps):
        print("TRAIN:", subprocess.list2cmdline(train), flush=True)
        print("TEST: ", subprocess.list2cmdline(eval_cmd), flush=True)
        if args.execute:
            subprocess.run(train, check=True)
            subprocess.run(eval_cmd, check=True)


if __name__ == "__main__":
    main()
