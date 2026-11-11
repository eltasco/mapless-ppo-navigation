# Experiments and benchmark protocol

## Published reference checkpoint

`models/navigation_ppo.zip` is the selected checkpoint. The public repository intentionally includes **only this policy**, rather than the full collection of intermediate training checkpoints. It was trained with PPO in the 2D environment described in the [technical brief](PROJECT_BRIEF.md).

During development, the experiments progressed through single-obstacle navigation, frame-history observations, mixed scene difficulty, risk-aware reward shaping, multi-obstacle fine-tuning and finally randomized 2–5-obstacle counts with a reduced learning rate. A larger learning rate in the final randomized-obstacle phase had previously degraded generalization; selecting a conservative fine-tuning rate yielded the published model.

## Local paired PPO vs. potential-field benchmark (11 November 2026)

The archived local run used:

```bash
python run.py benchmark --controllers both_legacy --difficulty hard --obstacles 2-5 --episodes 500
```

The legacy controller was forward-only and **not calibrated**; it is preserved under `--controllers potential_legacy`. The archived console transcript records the following values (500 episodes for each controller/scenario):

| Obstacles | Controller | Successes | Collisions | Timeouts | Wilson 95% success interval |
|--:|:--|--:|--:|--:|:--|
| 2 | PPO | 448 | 5 | 47 | 86.6–92.0% |
| 2 | Potential field | 85 | 0 | 415 | 14.0–20.5% |
| 3 | PPO | 471 | 18 | 11 | 91.8–95.9% |
| 3 | Potential field | 39 | 0 | 461 | 5.8–10.5% |
| 4 | PPO | 461 | 30 | 9 | 89.5–94.2% |
| 4 | Potential field | 17 | 0 | 483 | 2.1–5.4% |
| 5 | PPO | 437 | 58 | 5 | 84.2–90.0% |
| 5 | Potential field | 8 | 0 | 492 | 0.8–3.1% |

Across the equally weighted set of four scenario configurations, PPO reached 1,817/2,000 goals (90.85%), while the potential-field baseline reached 149/2,000 (7.45%). The latter recorded no collisions but timed out 1,851 times; PPO collided 111 times and timed out 72 times. These are **observational comparisons for this simulated distribution**, not an ablation establishing the causal benefit of observation history or the risk reward term.

For the exact provenance of the historical aggregate results and the limits of their verification, see [data provenance](DATA_PROVENANCE.md). The old comparison is **not** evidence of a superior learned controller relative to a calibrated classical planner. The revised baseline is evaluated separately after validation-only gain selection.

## Reported results

The following values were also recorded earlier during development (500 evaluation episodes per row). The latest user-run paired benchmark reproduces the four PPO `hard` rows in its console summary. The `mixed` row remains historical and has not been independently rerun.

| Difficulty | Obstacles | Success | Collision | Timeout |
|---|---:|---:|---:|---:|
| hard | 2 | 448 / 500 (89.6%) | 5 | 47 |
| hard | 3 | 471 / 500 (94.2%) | 18 | 11 |
| hard | 4 | 461 / 500 (92.2%) | 30 | 9 |
| hard | 5 | 437 / 500 (87.4%) | 58 | 5 |
| mixed | 5 | 427 / 500 (85.4%) | 57 | 16 |

For a computer-readable copy, see [`results/recorded_benchmarks.csv`](../results/recorded_benchmarks.csv).

## Reproduce the evaluation

```bash
python -m pip install -e ".[test]"
python run.py eval best hard 2-5 --episodes 500 --seed-start 60000 --csv results/evaluation.csv --json results/evaluation.json
```

The default obstacle radius is 0.35 m. All obstacle counts are evaluated over the same seed interval, but **different obstacle counts produce different scenario distributions**, so the four conditions are not identical maps with an incrementally added obstacle. The `hard_passable` setting enforces a gap constraint only for the first two obstacles; it does not certify full-world traversability. A radius change can alter both the obstacles' dimensions and the generator's feasible placement region.

To compare another checkpoint, use the exact same command and seeds with the path to that `.zip` file. Keep a record of the checkpoint SHA-256, simulator revision, versions, and the full CSV of episode outcomes. Do not infer hardware-level collision performance from simulator-only metrics.


## Additional experimental tools (not historical results)

- [Classical baseline, distribution shifts, Wilson confidence intervals and timeout traces](BASELINES_AND_ROBUSTNESS.md)
- [Multi-seed ablation protocol with frame-history and risk-weight controls](ABLATIONS.md)
- [`scripts/staged_training_example.sh`](../scripts/staged_training_example.sh) illustrates the curriculum structure but is not guaranteed to reproduce the released policy.

The published checkpoint records 190,464 cumulative PPO timesteps. The full training sequence cannot be recovered from it; original intermediate checkpoints and TensorBoard logs were not supplied. A documented reconstruction is not proof of identical training.
