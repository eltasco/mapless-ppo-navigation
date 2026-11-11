# Changelog

## 1.2.0 — 2026-11-11

- Documented the author's locally executed 2,000-scenario PPO-versus-potential-field benchmark (500 episodes for each of four obstacle counts, two controllers on paired seed ranges).
- Added the aggregate benchmark CSV/JSON, supplied console transcript and comparison figure. The original per-episode output was not provided and is not reconstructed.
- Added potential-field navigation, paired benchmarking, Wilson 95% intervals, and optional scenario/LiDAR stress tests and timeout tracing in the experimental toolkit.
- Added controlled ablation/training protocols, checkpoint integrity tests, and CI smoke coverage. No training ablation or out-of-distribution result is claimed without a measured run.

## 1.0.0 — 2026-11-11

- Packaged the Gymnasium navigation environment, procedural scenes, PPO utilities, 57D fixed-lag observations and Pygame viewer as an independently runnable research project.
- Included the original `navigation_ppo.zip` checkpoint with SHA-256 verification and the project methodology documentation.
