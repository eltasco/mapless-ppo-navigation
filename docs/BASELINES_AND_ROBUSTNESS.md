# Baselines, uncertainty and distribution shift

## 1. Paired classical baseline

The repository includes an **artificial potential-field controller**, inspired
by Khatib (1986), implemented in [`baselines.py`](../src/mapless_ppo/baselines.py).
It uses only the goal-relative features and 16 laser ranges used by the PPO
policy. The code does **not** implement DWA, mapping, global planning or an
optimized classical stack. Its fixed gains are declared in code and have not
been calibrated against the published test seeds.

Run both controllers on *exactly the same seeded episodes*:

```bash
python run.py benchmark --controllers both --difficulty hard --obstacles 2-5 --episodes 500 --seed-start 60000 --csv results/paired_benchmark.csv --json results/paired_benchmark.json
```

Outputs include per-episode outcomes, traveled distance, near-zero velocity
fraction, velocity reversals, closest obstacle clearance, and final goal
error. The JSON includes 95% Wilson score confidence intervals and paired
success/failure contingency counts (`ppo_only_success`, etc.).

**Interpretation:** the controller is a first baseline, not a claim of
state-of-the-art non-learning performance. PPO and the baseline receive the
same underlying scenario and initial measurement sequence (with identical
sensor-corruption seed where enabled). A high success rate alone does not
imply a collision-free or safe controller.

## 2. Out-of-distribution tests

The simulator supports controlled distribution shifts without modifying the
published checkpoint or default training/evaluation distribution:

```bash
# Positions of obstacles sampled along a wider strip; no midpoint anchor
python run.py benchmark --controllers both --difficulty hard --obstacles 5 --layout uniform --episodes 500 --csv results/uniform.csv --json results/uniform.json

# New goals (3.5–5.0 m) and independently generated obstacle positions
python run.py benchmark --controllers both --difficulty hard --obstacles 5 --layout long_goal --episodes 500 --csv results/long_goal.csv --json results/long_goal.json

# Noisy and intermittently missing range measurements
python run.py benchmark --controllers both --difficulty hard --obstacles 5 --lidar-noise 0.05 --dropout 0.05 --episodes 500 --csv results/noisy.csv --json results/noisy.json
```

* `original`: fixed historical obstacle placement distribution.
* `uniform`: goal still at 2–3 m, but obstacle positions independently sampled
  along a wider strip without a special midpoint obstacle.
* `long_goal`: target moves to 3.5–5 m, plus the alternative placement scheme.
* `--lidar-noise`: zero-mean Gaussian range noise, clipped to `[0, 3]` m.
* `--dropout`: independent per-ray probability of a max-range return.

These are **simulator stress tests**, not transfer-to-real validation. Walls,
non-circular obstacles, moving obstacles, odometry drift, and actuator delays
are **not** implemented. Range noise changes observations but not geometric
collision detection. No numerical OOD results are claimed until the commands
are executed with real dependencies.

Changing `--radius` on the original generator may change its *layout* as well
as its disk size; do not interpret that test as isolating radius alone.

## 3. Timeout and oscillation diagnostics

```bash
python run.py benchmark --controllers ppo --difficulty hard --obstacles 2 --episodes 500 --trace-csv results/timeout_traces.csv --trace-timeouts 5 --csv results/two_obstacles.csv --json results/two_obstacles.json
```

Timeout trajectories store step-level pose, requested normalized controls,
goal distance, and obstacle clearance. Investigate whether the policy spends
long periods near zero forward speed, flips the sign of linear velocity, or
oscillates around a pose. These are proxies, not definitive diagnoses. Plot them with:

```bash
python -m pip install -e ".[analysis]"
python scripts/plot_timeout.py results/timeout_traces.csv results/timeout_plot.png
```

The original **two-obstacle benchmark recorded 47 timeouts in 500 episodes**,
compared with 11 in the three-obstacle condition. Because the generators
differ by obstacle count, this does not establish a monotonic relationship
between scene difficulty and failure rate.

## 4. Uncertainty and provenance

Each row reports successes, collisions and timeouts separately. For `k`
successes out of `n` episodes, the reported Wilson score interval is a
binomial **95% interval**. It does not capture variation across independent
training seeds or distribution shifts. For a success rate near 90% with
`n=500`, a typical interval has a half-width of roughly 2.6 percentage points,
not 1.5 points. The paired table reports how often only one controller
succeeded on a common episode; it does not by itself imply statistical
significance.

To distinguish new evaluations from historical results, new JSON outputs
include `status=newly_measured_not_historical`, checkpoint SHA-256, settings and
seed start. The shipped `results/recorded_benchmarks.csv` is **historical**.

## Related work

- Khatib, O. (1986). [Real-Time Obstacle Avoidance for Manipulators and Mobile Robots](https://doi.org/10.1177/027836498600500106). *IJRR*, 5(1):90–98.
- Tai, L., Paolo, G., & Liu, M. (2017). [Virtual-to-real Deep Reinforcement Learning: Continuous Control of Mobile Robots for Mapless Navigation](https://doi.org/10.1109/IROS.2017.8202134). *IROS*.
- Schulman, J., et al. (2017). [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).
