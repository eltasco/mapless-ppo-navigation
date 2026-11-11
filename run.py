"""CLI entry point for mapless PPO navigation.

Examples:
    python run.py view best hard 5
    python run.py --help
    python run.py eval best hard 2-5 --episodes 500
    python run.py train scratch mixed 2-5 200000 models/new_ppo --lr 0.0003
"""
from pathlib import Path
import sys

# Permit running commands from a freshly cloned checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mapless_ppo.cli import main

if __name__ == "__main__":
    main()
