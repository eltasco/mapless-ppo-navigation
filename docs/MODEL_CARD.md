# Model Card — PPO Navigation Policy

## Identity and scope

- **File:** `models/navigation_ppo.zip`
- **Format:** Stable-Baselines3 PPO model archive (`.zip`, **do not extract**)
- **SHA-256:** `f6b44858775154b052c58286beeb713d40fd82bb08767d722f21462de2f16786`
- **Size:** 189,028 bytes
- **Task:** goal-directed local obstacle avoidance in an idealized 2D simulator
- **Intended input:** 57 float32 observations: three 19D LiDAR/goal frames at offsets `t−20`, `t−10`, `t`
- **Output:** two continuous controls, normalized to `[-1, 1]`

## Recorded architecture and optimization metadata

| Parameter | Saved value |
|---|---|
| Actor hidden layers | 64, 64 |
| Critic hidden layers | 64, 64 |
| Input / action dimensions | 57 / 2 |
| Saved learning rate | 0.0001 |
| Rollout length | 2048 |
| Batch size | 64 |
| PPO epochs | 10 |
| Discount `gamma` | 0.995 |
| GAE `lambda` | 0.95 |
| Clip range | 0.2 |
| Total timesteps stored in checkpoint | 190,464 |

The checkpoint's embedded `system_info.txt` records **Python 3.12.10, Stable-Baselines3 2.9.2, PyTorch 2.15.0+cpu, NumPy 2.5.4, Gymnasium 1.4.0**. These are environment metadata, not necessarily the minimum required versions.

## Verification status

The ZIP integrity test completed successfully; `policy.pth` was inspected with PyTorch's weights-only loading and has the expected actor and critic input sizes (**57**) and a two-dimensional action head. The refactored simulator was compared with the source on 240 matched deterministic scenes and matching action sequences. **An end-to-end Stable-Baselines3 replay was not available in this packaging environment**, so earlier success rates remain historical until explicitly rerun.

## Performance and limitations

Earlier local evaluations reported success rates from 87.4% to 94.2% on hard 2–5-obstacle scenes, with 500 episodes per configuration. Exact episode counts, outcomes and caveats are in [EXPERIMENTS.md](EXPERIMENTS.md).

The model is **not validated for use on real robots**. The simulator assumes perfect, sparse 2D range sensing and simplified kinematics. It has no obstacle motion, mapping, uncertainty estimation or external collision-prevention system. Use as a research/portfolio artifact rather than safety-critical autonomy software.

## Checkpoint verification

Run `python -m pytest tests/test_checkpoint.py` to check that the released archive is intact and matches the documented SHA-256 and model configuration. The checkpoint is an SB3 archive, not a standalone PyTorch state dict.
