# Classical baselines and robustness protocol

## Two potential-field versions

`potential_legacy` preserves the original forward-only, uncalibrated controller that produced the historical 7.45% aggregate success rate. It is provided **for reproducibility, not as a strong competitive baseline**. Its high timeout rate is consistent with potential-field stagnation; it is not proof of safety.

`potential` is a revised controller with **capped repulsion**, a deterministic **tangential bypass component**, and **signed backward control**. It is still a heuristic reactive controller, not Dynamic Window Approach or a global planner. Its default parameters have **not** been validated and are explicitly labeled *untuned* by `benchmark`.

### Parameter selection on disjoint seeds

```bash
python run.py tune-baseline --difficulty hard --obstacles 2-5 --seed-start 30000 --episodes 30 --output configs/potential_validated.json
```

This searches 18 parameter combinations: repulsive gains 0.08, 0.16, 0.32; tangential gains 0.0, 0.45, 0.9; reverse allowed or disabled. Each candidate receives identical seeded 2–5-obstacle scenes. The score prioritizes goal reaching with an explicit collision penalty and a weak step-count tie-breaker. The output JSON records **every candidate and all validation counts**. Tuning refuses overlap with the published 60,000–60,499 held-out test range.

```bash
python run.py benchmark --controllers both --baseline-config configs/potential_validated.json --difficulty hard --obstacles 2-5 --episodes 500 --seed-start 60000 --csv results/paired_benchmark.csv --json results/paired_benchmark.json
```

To reproduce the original untuned controller *without changing its gains*:

```bash
python run.py benchmark --controllers both_legacy --difficulty hard --obstacles 2-5 --episodes 500
```

Only **measured, held-out** comparisons should be added to benchmark tables. This release contains no such measurements for `potential` or `potential_tuned`. See [data provenance](DATA_PROVENANCE.md) for the historical transcript.

## Stress tests and history ablations use the same evaluator

`eval` and `benchmark` use the shared `run_episode` implementation. Both support:

- `--history single` (19D) or `--history stack` (57D), with checkpoint shape verification;
- `--layout uniform` (no forced midpoint obstacle) or `long_goal` (3.5–5 m goal plus alternate layout);
- `--lidar-noise` (Gaussian meters) and `--dropout` (missing rays represented as max range);
- `--observation-scale fixed` **only with an appropriate newly trained checkpoint and sidecar**.

These tests perturb static circular scenes and/or simulated measurements. They **do not** introduce walls, dynamic obstacles, 3D occlusions or noisy localization. The baseline remains reactive and can fail on layouts outside the training family.

```bash
python run.py eval best hard 5 --layout uniform --episodes 500 --csv results/uniform.csv
python run.py eval best hard 5 --layout long_goal --episodes 500 --csv results/long_goal.csv
python run.py eval best hard 5 --lidar-noise 0.05 --dropout 0.05 --episodes 500 --csv results/noisy.csv
```

Do not interpret a changed obstacle radius as *only* a change in size: the scene generator may alter its placement constraints when radius changes.

## Timeout diagnosis

The historical PPO run recorded 47/500 timeouts at two obstacles, versus 11/500 at three. This suggests an interesting failure mode, **not** a demonstrated causal mechanism. Generate a new 500-episode trace file and inspect low-speed fractions, sign flips, remaining goal distance and representative trajectories:

```bash
python run.py benchmark --controllers ppo --obstacles 2 --episodes 500 --trace-csv results/timeout_traces.csv --trace-timeouts 10 --csv results/two_obstacles.csv
python scripts/timeout_summary.py results/two_obstacles.csv --obstacles 2 --output results/timeout_summary.json
python -m pip install -e ".[analysis]"
python scripts/plot_timeout.py results/timeout_traces.csv results/timeout_plot.png
```

The diagnostic script provides **descriptive statistics**, not a claim that all timeouts are oscillations. When recording failures, check the scene generation family and the seed range.
