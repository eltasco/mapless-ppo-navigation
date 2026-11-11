"""Shared rollout, scaling contract and validation-config checks."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from mapless_ppo.baselines import PotentialFieldController, load_baseline_config
from mapless_ppo.benchmark import make_benchmark_env
from mapless_ppo.experiments import check_checkpoint_observations
from mapless_ppo.rollouts import run_episode
from mapless_ppo.tuning import tune_baseline
from mapless_ppo.analysis import analyze
from mapless_ppo.artifacts import verify


def test_fixed_scale_keeps_shape_and_reproducibility():
    a = make_benchmark_env(2, "hard", "original", 0, 0, "stack", "none")
    b = make_benchmark_env(2, "hard", "original", 0, 0, "stack", "fixed")
    try:
        raw, _ = a.reset(seed=100)
        scaled, _ = b.reset(seed=100)
        assert raw.shape == scaled.shape == (57,)
        np.testing.assert_allclose(raw.reshape(3, 19)[:, 0] / 5,
                                   scaled.reshape(3, 19)[:, 0])
        np.testing.assert_allclose(raw.reshape(3, 19)[:, 3:] / 3,
                                   scaled.reshape(3, 19)[:, 3:])
    finally:
        a.close()
        b.close()


def test_shipped_checkpoint_rejects_wrong_preprocessing():
    check_checkpoint_observations("models/navigation_ppo.zip", "stack", "none")
    with pytest.raises(ValueError):
        check_checkpoint_observations("models/navigation_ppo.zip", "single", "none")
    with pytest.raises(ValueError):
        check_checkpoint_observations("models/navigation_ppo.zip", "stack", "fixed")


def test_tuned_config_round_trip(monkeypatch, tmp_path):
    import mapless_ppo.tuning as tuning
    monkeypatch.setattr(tuning, "candidate_parameters", lambda: iter([PotentialFieldController()]))
    output = tmp_path / "tuned.json"
    result = tune_baseline(stage="hard", counts=[2], seed_start=30_000, episodes=1,
                           output=str(output))
    assert result["kind"] == "validated_potential_field_v2"
    assert load_baseline_config(output) == PotentialFieldController()


def test_episode_result_deterministic_by_seed():
    env = make_benchmark_env(2, "hard", "uniform", 0.02, 0.03, "single")
    ctrl = PotentialFieldController()
    try:
        a, _ = run_episode(env, ctrl, 31_000, None)
        b, _ = run_episode(env, ctrl, 31_000, None)
        assert a == b
    finally:
        env.close()


def test_timeout_summary_requires_actual_rows():
    rows = [{"controller": "ppo", "obstacles": "2", "outcome": "timeout",
             "seed": "7", "low_speed_fraction": "0.8", "velocity_sign_flips": "12",
             "end_goal_distance_m": "2.1"}]
    result = analyze(rows, 2)
    assert result["timeout_seeds"] == [7]
    assert result["low_speed_fraction_among_timeouts"]["mean"] == pytest.approx(0.8)
    with pytest.raises(ValueError):
        analyze(rows, 3)


def test_episode_artifacts_checker_reconciles_counts(tmp_path):
    csvpath = tmp_path / "paired.csv"
    jsonpath = tmp_path / "paired.json"
    with csvpath.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=["controller", "stage", "obstacles", "seed", "outcome"])
        writer.writeheader()
        writer.writerow({"controller": "ppo", "stage": "hard", "obstacles": 2, "seed": 10, "outcome": "success"})
    jsonpath.write_text(json.dumps({"summaries": [{"controller": "ppo", "stage": "hard",
        "obstacles": 2, "episodes": 1, "success": 1, "collision": 0, "timeout": 0}]}))
    assert verify(csvpath, jsonpath)["rows"] == 1
    jsonpath.write_text(json.dumps({"summaries": [{"controller": "ppo", "stage": "hard",
        "obstacles": 2, "episodes": 1, "success": 0, "collision": 1, "timeout": 0}]}))
    with pytest.raises(ValueError):
        verify(csvpath, jsonpath)
