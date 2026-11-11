# Technical Brief: Mapless Local Navigation with PPO

## Problem and motivation

A mobile agent must reach a goal while avoiding static obstacles, without access to a precomputed occupancy map. The learned policy operates from goal-relative coordinates and sparse, local range information. The environment is purposefully simple enough to train and inspect on CPU, while exposing meaningful control and partial-observability challenges: limited angular resolution, obstacle clearance, reverse motion and local oscillation.

## Simulator and task definition

- **Robot:** disk approximation of radius 0.20 m, planar pose `(x, y, heading)`, kinematic control integration at 30 Hz.
- **Environment:** procedurally generated goal and one to five static circular obstacles, with difficulty-dependent offsets. The default obstacle radius is 0.35 m.
- **Goal:** arrive within 0.25 m of target before the 900-step horizon, without contacting an obstacle.
- **Inputs:** goal distance, sine and cosine of the relative goal bearing, and 16 idealized 360° range rays of up to 3.0 m.
- **Frame history:** observations at `t−20`, `t−10` and `t` concatenated into 57 float32 inputs. The earliest observation pads the history at reset.
- **Policy outputs:** continuous `(a_v, a_ω)` in `[-1, 1]^2`, scaled to signed velocity ±0.6 m/s and yaw rate ±1.5 rad/s.

The simulator owns the true obstacle positions, but the PPO policy receives only simulated sensor and goal-relative data. This distinction is critical to the mapless claim.

## Reinforcement-learning objective

The scalar reward comprises goal-distance progress, a time cost, a near-obstacle proximity cost, a closing-speed risk cost and terminal rewards:

$$
\begin{aligned}
r_t &= (d_{t-1}-d_t) - 0.1\Delta t \\
&\quad - 0.8p(c_t)\Delta t \\
&\quad - 1.5\max_i\left[p(c_{i,t})\left(\frac{\max(0,\hat{u}_{i,t}\cdot v_t)}{v_{\max}}\right)^2\right]\Delta t \\
&\quad + 10\mathbf{1}_{\mathrm{goal}} - 10\mathbf{1}_{\mathrm{collision}}.
\end{aligned}
$$

Here `p(c) = clip((0.35 - c) / 0.35, 0, 1)`, `c` is the clearance between robot and obstacle boundaries, and `û` points from the robot toward an obstacle. The closing-speed penalty is active only for **positive velocity toward nearby obstacles**; reversing away from one is not penalized by that term. The implementation uses post-step heading for risk projection, preserving the convention used during training; translation uses pre-step heading. Terminal rewards have magnitude ±10.

## Policy, optimization, evaluation

The shipped checkpoint is a PPO MLP policy with distinct 64–64 actor and critic networks. Its metadata stores 2,048 rollout steps, batch size 64, ten optimization epochs, discount factor 0.995, GAE parameter 0.95 and PPO clipping coefficient 0.2. The model stores a learning rate of 1e-4, reflecting fine-tuning; this does not imply it was trained from scratch at that rate.

Evaluation uses `model.predict(..., deterministic=True)` and records success, collision or timeout for each episode, as well as reward and duration. The default seed range is 60000–60499 per configuration, enabling comparisons on fixed procedural scenes. The Pygame visualizer helps inspect failures that a scalar success rate can conceal, such as oscillations and dangerously close passes.

## Engineering decisions and limitations

**Why not train directly on physical hardware?** The 2D model permits rapid reward iteration and a clear algorithmic baseline. No transfer to hardware is asserted.

**Why history instead of an RNN?** Fixed-delay frame stacking provides time context with an ordinary feed-forward MLP and is easy to inspect. It is not a general memory mechanism.

**Why risk-aware shaping?** Distance-to-goal alone can reward trajectories that pass too close to obstacles. A proximity-weighted positive closing-speed cost discourages aggressive approach and specifically captures reverse-motion hazards.

**What this does not model:** moving obstacles, stochastic LiDAR errors, dynamics/traction, actuation delay, SLAM, global path planning, 3D geometry, end-to-end safety and physical robot operation. The scene generator also does not guarantee a globally collision-free route for every sampled layout.

For checkpoint provenance and precise reproducibility caveats, see [MODEL_CARD.md](MODEL_CARD.md) and [EXPERIMENTS.md](EXPERIMENTS.md).


## Comparative controls and failure analysis

The repository now provides an artificial potential-field baseline on paired deterministic scenes, as well as non-midpoint obstacle layouts, longer goals, and measurement noise/dropout. These are *experiment harnesses*, not evidence of better performance until run and recorded. See [Baselines and robustness](BASELINES_AND_ROBUSTNESS.md) and [Ablation protocol](ABLATIONS.md).
