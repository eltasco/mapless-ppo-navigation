# Controlled training ablations

The included `navigation_ppo.zip` is a frozen reference checkpoint. Removing
history or changing the training reward **cannot** be tested by simply editing
its inputs or recompiling its source: a fair ablation requires independent
policies trained under the same budget and evaluated on held-out seeds.

These examples are **protocols, not precomputed experiments**. They use one
seed to illustrate the setup. For a more credible study, repeat with at least
three independent training seeds (`--seed 42`, `43`, `44`, etc.) and report
mean and standard deviation. Do not select the best run after inspecting test
seeds.

## Three single-budget configurations

```bash
# Default: 57-D fixed-lag observation and closing-speed risk penalty
python run.py train scratch mixed 2-5 200000 models/ablation_full_seed42 --seed 42 --history stack --risk-weight 1.5 --lr 0.0003 --eval-every 10000 --tensorboard-log runs

# No history: only the current 19-D observation; requires new network weights
python run.py train scratch mixed 2-5 200000 models/ablation_no_history_seed42 --seed 42 --history single --risk-weight 1.5 --lr 0.0003 --eval-every 10000 --tensorboard-log runs

# No closing-speed risk penalty; geometry and observations unchanged
python run.py train scratch mixed 2-5 200000 models/ablation_no_risk_seed42 --seed 42 --history stack --risk-weight 0 --lr 0.0003 --eval-every 10000 --tensorboard-log runs
```

Install logging with `python -m pip install -e ".[test,logging]"` and inspect
training progress with `tensorboard --logdir runs`. The validation callback
saves periodic evaluation traces and a validation-best checkpoint; held-out
episodes starting from seed 60000 should be reserved for the final comparison.

## Compare using identical evaluation seeds

```bash
python run.py eval models/ablation_full_seed42 hard 5 --episodes 500 --seed-start 60000 --history stack --csv results/ablation_full.csv
python run.py eval models/ablation_no_history_seed42 hard 5 --episodes 500 --seed-start 60000 --history single --csv results/ablation_no_history.csv
python run.py eval models/ablation_no_risk_seed42 hard 5 --episodes 500 --seed-start 60000 --history stack --csv results/ablation_no_risk.csv
```

Compare success, collision, timeout and training efficiency. The original
checkpoint's historical path differs from the above flat 200k-step curriculum,
so these ablations **must not** be labeled as an exact reproduction of the
published checkpoint. Sensitivity to observation normalization (e.g.
`VecNormalize`) would also require *retraining* and versioned normalization
statistics; never apply a newly fitted normalizer to the bundled checkpoint.

## Checkpoint provenance and limitations

The serialized reference checkpoint records **190,464 cumulative timesteps**,
`learning_rate=0.0001`, and 64×64 actor/critic MLP architectures. Intermediate
checkpoints, original TensorBoard event files and a complete original training
configuration manifest were not supplied. The documented development stages
(1 obstacle, frame history, mixed layouts, risk penalty, 3 obstacles, 2–5
obstacles at lower LR) are based on contemporaneous experiment notes, not
reconstructable from the final ZIP alone. Consequently **bitwise or statistical
reproduction of the historical training trajectory is not claimed**.

The simulator still uses ideal unnormalized metric distances. Any change to
reward timing (post-update heading), range ordering or normalization can break
comparison with the frozen policy. These are compatibility constraints, not
necessarily best engineering choices for training a new system.
