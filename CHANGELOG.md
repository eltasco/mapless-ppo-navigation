# Changelog

## 1.3.0 — Benchmark methodology and evaluation tooling

- Preserved the original untuned potential-field controller as `potential_legacy`, distinct from a configurable revised field with bounded repulsion, tangential steering and reverse motion.
- Added explicit validation-only parameter selection on seed ranges disjoint from published tests. **No calibrated-baseline performance is claimed without a measured run.**
- Centralized episode rollout metrics; `eval` and `benchmark` support 19D and 57D policies, alternative scene layouts, LiDAR noise/dropout and optional fixed scaling for new trained models.
- Expanded validation across obstacle counts and fixed seeds, added a three-seed ablation runner, timeout summarizer and episode-level result checker.
- Reframed the old weak-field comparison as exploratory, organized data provenance and strengthened baseline behavior and preprocessing tests.
- Preserved the shipped PPO checkpoint and its SHA-256, with no changes to the original policy's learned weights.

## 1.2.0 — Initial public repository packaging

- Packaged the reference PPO checkpoint, 2D simulator, baseline, evaluation tools, Pygame viewer and technical notes.
- The public repository was assembled in a concentrated documentation/refactoring session. Same-day commits are **not** a reconstruction of the earlier development chronology; the original staged training happened before the public history.
- A few generated smoke-test output files were briefly committed while preparing the repository, then removed. They were **not used as published benchmark evidence**. The Git history has not been rewritten.