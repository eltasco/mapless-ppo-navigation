# Benchmark results and provenance

## Locally measured PPO–Potential Field comparison

- [`paired_benchmark_summary.csv`](paired_benchmark_summary.csv): eight rows (PPO and the classical baseline, 2–5 obstacles), each with success, collision, timeout and reported Wilson interval.
- [`paired_benchmark_summary.json`](paired_benchmark_summary.json): structured experiment settings, the checkpoint hash, aggregated outcomes and provenance.
- [`reported_benchmark_output.txt`](reported_benchmark_output.txt): verbatim benchmark lines supplied by the project author on 11 November 2026.
- [`recorded_benchmarks.csv`](recorded_benchmarks.csv): older PPO-only development records, including a separate `mixed` condition.

The author ran the benchmark locally and posted its **console summary**. These files are curated transcriptions, not freshly rerun results on CI. In particular **the original per-episode `paired_benchmark.csv` and `.json` outputs were not provided**. Thus, individual trajectories, paired outcome contingency counts and paired hypothesis tests cannot be reconstructed.

The four `hard` PPO measurements match the previously recorded PPO-only benchmark rows. Across all four settings the PPO success rate is 90.85% versus 7.45% for the baseline. The potential-field controller's zero collisions accompany 92.55% timeouts, so do not interpret it as a safe or effective planner.

## Reproduce on your machine

```bash
python -m pip install -e ".[test]"
python run.py benchmark --controllers both --difficulty hard --obstacles 2-5 --episodes 500 --csv results/paired_benchmark.csv --json results/paired_benchmark.json
```

Unlike the published summary, a fresh run produces **one row per episode**, paired by seed in the original generator, and includes paired success/failure counts in JSON. Generated outputs are ignored by Git by default; the reviewed summary files are explicitly included.
