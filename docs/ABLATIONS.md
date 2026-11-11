# Controlled PPO ablation protocol

The frozen reference checkpoint uses a 57-element history (`stack`) and a risk reward coefficient of 1.5. A change in input history, risk shaping or observation normalization **requires retraining**, not simply toggling an inference flag.

## Three variants × three training seeds

Run the generator in dry-run mode to inspect all nine jobs before committing CPU time:

```bash
python scripts/run_ablations.py
python scripts/run_ablations.py --execute
```

Each job trains 200,000 nominal PPO steps on the same procedural family with seeds 42, 43, 44, then evaluates on the held-out seed range 60,000–60,499. The three variants are: **full** (57D, risk weight 1.5), **no history** (19D, weight 1.5) and **no risk term** (57D, weight 0). The examples are new protocols and **not a recreation of the staged training schedule** that produced the shipped checkpoint.

The optional `--tensorboard-log runs` enables PPO training curves (install `.[logging]`). `--eval-every 10000` selects validation-best checkpoints with **50 episodes for each obstacle count** on fixed seeds from 20,000, not a single noisy 10-episode five-obstacle validation. The command also retains the final checkpoint; both should be analyzed with transparent selection rules.

## Evaluate any variant on the same scenarios

```bash
python run.py benchmark --controllers ppo --model models/ablation_no_history_seed42.zip --history single --difficulty hard --obstacles 2-5 --episodes 500
python run.py eval models/ablation_no_history_seed42.zip hard 5 --history single --layout uniform --episodes 500
```

Both evaluation commands now share exactly the same episode execution loop and accept the same stress settings.

## Optional feature scaling for new training

`--observation-scale fixed` rescales goal distance by 5 m and LiDAR distances by the 3 m sensor maximum, preserving sine/cosine bearing. It is **not** running-statistics `VecNormalize`; there are no learned statistics to serialize. Checkpoint JSON sidecars record the choice and validation rejects mismatched settings. The released PPO weights must use raw, unnormalized inputs.

## What cannot be concluded from these scripts alone

No ablation success rate, improvement, training-seed variance or OOD robustness claim is made without running the jobs and publishing their episode-level data. See [data provenance](DATA_PROVENANCE.md).
