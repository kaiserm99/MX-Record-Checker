# Stick Balancer

An AI that teaches itself to balance a stick on a cart, then a stick with one joint
(double inverted pendulum), then two joints (triple), and so on.  Everything is
small, explicit and readable:

| File | Purpose |
|---|---|
| `physics.py` | N-link cart-pendulum dynamics in mass-matrix form `M(q) q'' = f(q, q', u)`, with realistic air drag and bearing/rail friction, RK4 integration, energy and geometry helpers. Pure NumPy. |
| `test_physics.py` | Proves the derivation: matches Gymnasium's CartPole-v1 equations to 1e-10 for one link, RK4 energy drift shrinks like h^4 for 1-4 links, rod drag torque matches its closed form, and energy never increases with dissipation on. |
| `env.py` | The Gymnasium environment (observation, action, reward, termination). |
| `recipes.py` | Every tuning number per number of links, in one place, with the reasoning. |
| `train.py` | Trains PPO (or SAC) with Stable-Baselines3: normalisation, parallel envs, periodic evaluation, early stop when solved. |
| `evaluate.py` | Runs a trained policy, prints returns, exports a trajectory for the viewer. |
| `make_report.py` + `report_template.html` | Builds `report.html`: an animation of each trained agent, learning curves, the maths, and the research notes. |
| `run_all.sh` | Trains and evaluates the whole ladder 1 -> 2 -> 3 links. |

## Quick start

```bash
pip install -r requirements.txt
python test_physics.py                     # sanity-check the simulator
python train.py --links 1                  # ~1 minute on a laptop CPU
python train.py --links 2                  # double inverted pendulum
python train.py --links 3 --timesteps 5e6  # triple inverted pendulum
python evaluate.py --links 2 --export runs/ppo_2links/trajectory.json
python make_report.py && open report.html
```

Change the physics in `physics.py`, the reward in `env.py`, the hyperparameters in `recipes.py`.
Add `--algo sac` for Soft Actor-Critic (more sample-efficient, slower per step).

## How it works, in one paragraph each

**Physics.** Coordinates are the cart position `x` and the absolute angle of every
link from vertical, `theta_i`.  Each link is a uniform rod.  Using d'Alembert's
principle the equations of motion are `M(q) q'' = u e0 + sum_i J_i^T (F_grav_i -
m_i Jdot_i q') - damping`, with the mass matrix `M = m0 e0 e0^T + sum_i (m_i J_i^T J_i
+ I_i e_i e_i^T)`.  `J_i` is the Jacobian of the centre of mass of link `i`, which is
trivial to write down because the position of that centre is a sum of `l_j sin/cos
theta_j` terms.  We solve the linear system numerically every sub-step, so the same
code handles any number of links.  Integration is classic RK4 with 5 ms sub-steps.

**Realism.** By default the stick lives in air and turns on real bearings:
quadratic air drag on every link (cylinder cross-flow, C_d 1.1, 2 cm dowel,
integrated along the rod with Gauss-Legendre quadrature), air drag on the cart,
viscous + Coulomb bearing friction in each hinge (0.002 N m s/rad, the value
measured on TU Wien's triple pendulum, plus 1e-4 N m dry friction) and Coulomb
rail friction on the cart (mu = 0.01).  Coulomb terms are tanh-smoothed so RK4
stays smooth.  Every coefficient is a field of `CartPendulumParams`; use
`physics="ideal"` in the env for the frictionless textbook model, or
`physics_overrides={"link_diameter": 0.03}` to tweak one thing.

**Task.** The agent sees `[x, x', theta_1..N, theta'_1..N]` and outputs one number in
[-1, 1], scaled to a horizontal force on the cart.  Reward is 1 per step alive minus
tiny quadratic penalties on cart offset, lean, and force (max 1000 per 20 s episode).
The episode ends when the cart leaves +/-2.4 m or a link leans past the limit.

**Learning.** PPO from Stable-Baselines3 with `VecNormalize` (observation and reward
scaling), 4 subprocess environments, linearly decaying learning rate and clip range,
an `EvalCallback` that runs 10 deterministic episodes every 20k steps, and a
`StopTrainingOnRewardThreshold` at 980 so training stops when the stick is actually
balanced.  The saved `vecnormalize.pkl` is part of the model: load it at test time.

## Research summary

See `report.html` (section "What the research says") for the full notes with links.
The short version:

* **PPO** solves CartPole in ~1e5 steps and MuJoCo's InvertedDoublePendulum in ~1e6.
  **SAC/TD3** need 10-50x fewer environment steps but more compute per step.  DQN is
  limited to discrete actions and does not scale to multi-joint sticks.
* **More joints is much harder** because the upright equilibrium has N unstable modes,
  the fastest of which gets faster with N and shorter links; one force has to
  stabilise all of them (underactuation N), so the policy needs precise velocities
  and high control bandwidth.  The recipes raise force authority and the lean limit
  with N.
* **Practices that matter**: normalise observations and rewards; alive bonus plus small
  quadratic costs (MuJoCo's InvertedDoublePendulum shape); terminate on lean so bad
  episodes are short; integrate accurately (RK4 with sub-steps); evaluate
  deterministically and stop on a threshold.
* **Going further**: swing-up (sin/cos observations, tip-height reward), curriculum
  on initial perturbation, domain randomisation for sim-to-real, MuJoCo's
  `InvertedDoublePendulum-v5` for comparison with published scores.
