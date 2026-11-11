# Experiments and benchmark protocol

## Published reference checkpoint

`models/navigation_ppo.zip` is the selected checkpoint. The public repository intentionally includes **only this policy**, rather than the full collection of intermediate training checkpoints. It was trained with PPO in the 2D environment described in the [technical brief](PROJECT_BRIEF.md).

During development, the experiments progressed through single-obstacle navigation, frame-history observations, mixed scene difficulty, risk-aware reward shaping, multi-obstacle fine-tuning and finally randomized 2–5-obstacle counts with a reduced learning rate. A larger learning rate in the final randomized-obstacle phase had previously degraded generalization; selecting a conservative fine-tuning rate yielded the published model.

## Reported results

The following results were measured during development (500 evaluation episodes per row). They are **historical records** and should be compared against a new evaluation performed on this cleaned code, not described as an automated CI benchmark.

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

## Reproducing the evaluation protocol

Run `python run.py eval best hard 2-5 --episodes 500` from the repository root to evaluate the bundled checkpoint across the hard obstacle-count settings. The newly generated measurements should be reported separately from the historical reference table until seed ranges and environment revision are confirmed to match.
