"""Regression tests for CLI parsing, option validation and checkpoint aliases."""

import pytest

pytest.importorskip("gymnasium")

from mapless_ppo.cli import build_parser
from mapless_ppo.experiments import parse_counts, parse_stages
from mapless_ppo.scenarios import parse_radius


def test_cli_arguments_parse():
    parser = build_parser()
    args = parser.parse_args(["eval", "models/navigation_ppo.zip", "hard", "5", "0.50"])
    assert (args.model, args.difficulty, args.obstacles, args.radius) == (
        "models/navigation_ppo.zip", "hard", "5", "0.50"
    )
    assert args.episodes == 500
    train = parser.parse_args([
        "train", "best", "mixed", "2-5", "10000", "models/ppo_test_v3", "--lr", "0.0001",
    ])
    assert train.lr == pytest.approx(1e-4)


def test_parsers():
    assert parse_counts("2-5") == [2, 3, 4, 5]
    assert parse_counts("2,4,5") == [2, 4, 5]
    assert parse_stages("hard_passable") == ["hard_passable"]
    assert parse_radius("original") is None
    assert parse_radius("0.25-0.5") == (0.25, 0.5)
    with pytest.raises(ValueError):
        parse_counts("0")
    with pytest.raises(ValueError):
        parse_radius("2.0")


def test_default_checkpoint_alias():
    from mapless_ppo.experiments import actual_zip, model_path
    assert actual_zip(model_path("best")).as_posix() == "models/navigation_ppo.zip"


def test_benchmark_and_ablation_options():
    parser = build_parser()
    bench = parser.parse_args(["benchmark", "--controllers", "both", "--layout", "uniform", "--lidar-noise", "0.05"])
    assert bench.layout == "uniform"
    assert bench.lidar_noise == pytest.approx(0.05)
    train = parser.parse_args(["train", "scratch", "hard", "2-5", "10000", "models/new", "--history", "single", "--risk-weight", "0", "--eval-every", "1000"])
    assert train.history == "single" and train.risk_weight == 0 and train.eval_every == 1000


def test_new_cli_options():
    parser = build_parser()
    e = parser.parse_args(["eval", "best", "hard", "5", "--history", "single", "--layout", "uniform", "--lidar-noise", "0.04"])
    assert e.history == "single" and e.layout == "uniform" and e.lidar_noise == 0.04
    b = parser.parse_args(["benchmark", "--controllers", "potential_legacy", "--history", "single"])
    assert b.controllers == "potential_legacy" and b.history == "single"
    t = parser.parse_args(["tune-baseline", "--seed-start", "30000", "--episodes", "20"])
    assert t.episodes == 20 and t.seed_start == 30000
    tr = parser.parse_args(["train", "scratch", "mixed", "2-5", "10000", "models/example", "--validation-episodes", "50", "--observation-scale", "fixed"])
    assert tr.validation_episodes == 50 and tr.observation_scale == "fixed"


def test_historical_paired_controller_alias():
    args = build_parser().parse_args(["benchmark", "--controllers", "both_legacy"])
    assert args.controllers == "both_legacy"
