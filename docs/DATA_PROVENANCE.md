# Data provenance and interpretation

## Available material

The repository ships the unchanged reference checkpoint, [`models/navigation_ppo.zip`](../models/navigation_ppo.zip), which is integrity checked by SHA-256. It also retains a console transcript and aggregate counts from an in-distribution PPO vs. **historical untuned potential-field** evaluation performed locally on 11 November 2026:

- [`results/reported_benchmark_output.txt`](../results/reported_benchmark_output.txt): recorded console lines.
- [`results/paired_benchmark_summary.csv`](../results/paired_benchmark_summary.csv): aggregate counts and Wilson intervals transcribed from those lines.
- [`results/paired_benchmark_summary.json`](../results/paired_benchmark_summary.json): structured aggregate settings and counts.
- [`results/recorded_benchmarks.csv`](../results/recorded_benchmarks.csv): an earlier PPO-only experiment summary.

The **per-episode raw files** (`paired_benchmark.csv` and `paired_benchmark.json`) from that historical run were not included in the materials accessible for this release. The aggregate rows **do not prove which seeds succeeded for each controller** and are not sufficient to compute the paired contingency table, McNemar test or error traces. We have not manufactured missing episodes or claimed the raw data were independently verified.

## How to obtain and check new results

The benchmark command writes real per-episode CSV and metadata JSON. Keep these files, the checkpoint hash, command-line flags, seeds and simulator revision together:

```bash
python run.py benchmark --controllers both --baseline-config configs/potential_validated.json --difficulty hard --obstacles 2-5 --episodes 500 --csv results/paired_benchmark.csv --json results/paired_benchmark.json
python scripts/check_episode_artifacts.py results/paired_benchmark.csv results/paired_benchmark.json
```

The checker validates one row per controller/scenario/seed and reconciles outcomes with the exported summary. It cannot prove simulation fidelity or that the baseline was well designed, but it prevents passing a manually typed aggregate table off as episode-level evidence.

## Interpreting uncertainty

Wilson confidence intervals express **episode-sampling uncertainty for a fixed policy and scene distribution**. They do not capture the variance of different PPO training seeds, differences in generalization under distribution shifts, or real-robot safety. Both controllers face identical *scene seeds* in paired comparisons, but their action-driven measurements differ across trajectories. The historical 7.45% field result was produced with an **untuned, forward-only controller**, and therefore cannot support claims about optimized classical planning.
