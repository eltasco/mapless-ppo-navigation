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
