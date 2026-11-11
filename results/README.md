# Evaluation results

`recorded_benchmarks.csv` contains historical evaluation summaries of the shipped PPO model. The numbers are labeled as previously recorded results, not as a newly reproduced CI test.

Generate your own measurements with:

```bash
python run.py eval best hard 2-5 --episodes 500 --csv results/evaluation.csv --json results/evaluation.json
```

The CSV records each episode's seed, environment configuration, outcome and reward; JSON contains the aggregate results. These generated files are excluded from version control by default.

**Interpretation:** The included benchmark summarizes simulation performance. Newly generated evaluations should retain the selected checkpoint, seed range and scenario settings alongside the exported files.
