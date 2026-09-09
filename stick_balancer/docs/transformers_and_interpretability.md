# Would a transformer make sense here, and is there a link to LLM interpretability?

Assessment written 2026-09-09 after a literature pass and one experiment on the trained
agents in this repo (`probe_linear.py`).

## Short answer

* **As a drop-in policy for the tasks as they stand: no.**  The state is fully observed and
  the physics is fixed, so the Markov state is sufficient and a two-layer MLP is the right
  model class.  The evidence says a transformer adds cost without benefit here: RvS shows a
  two-layer MLP matches Decision Transformers on the standard offline benchmarks
  (https://arxiv.org/abs/2112.10751), POPGym finds GRUs beating attention models on most
  partially observed RL tasks (https://arxiv.org/abs/2303.01859), and transformers for
  online continuous control are reported to be architecture-sensitive
  (https://arxiv.org/abs/2510.13367).
* **As a research instrument: yes, if you change the task so that memory is required.**
  Randomise the physics per episode (mass, length, drag, friction) and hide the
  parameters, or hide the velocities.  Then a history-conditioned policy must do system
  identification and filtering *in context*, which transformers provably can do
  (Goel & Bartlett: a causal transformer can implement the Kalman filter and hence LQG,
  https://arxiv.org/abs/2312.06937; trained transformers match KF/EKF in context,
  https://arxiv.org/abs/2410.16546; Forgione's in-context sysid,
  https://github.com/forgi86/sysid-transformers).  That variant is where the
  interpretability connection lives.

## What the probe found about the current MLPs

`probe_linear.py` linearises the physics at upright, solves the LQR (the textbook optimal
linear controller), and compares its gain vector with the Jacobian of the trained policy.

| links | cosine(LQR gain, network gain) | network's linearisation stabilising? | linear fit R² on visited states | force saturation |
|---|---|---|---|---|
| 1 | 0.97 | marginal (radius 1.006) | 0.96 | none (±6 of 10 N) |
| 2 | 0.996 | no (1.21) | 0.998 | hits ±20 N |
| 3 | 0.99 | no (3.5) | 0.76 | hits ±30 N |

Reading: the networks point their gain in almost exactly the LQR direction, but with gains
far larger than any stabilising linear law, plus a non-zero force bias at exact upright.
They are not linear controllers; they are *saturating* controllers that use the LQR
direction to decide which way to slam the cart and rely on the force clip.  For 3 links
the policy is only 76% linear on its own trajectories.  This is the first thing an
interpretability study of these policies would have to explain (why does PPO converge to
"bang-bang along the LQR direction"?), and it is not something the literature has looked at:
no paper probes a trained PPO cart-pole policy for its local gain (adjacent: policy gradient
provably approaches LQR on the linearisation, https://proceedings.neurips.cc/paper_files/paper/2021/file/f4f6dce2f3a0f9dada0c2b5b66452017-Supplemental.pdf;
QRnet shows accurate networks can still fail to stabilise locally, https://arxiv.org/abs/2205.00394).

A second honest finding from the same session: the 3-link agent that scored 997.8 over
10 evaluation episodes scores 741 ± 426 over 30, i.e. it still fails about a third of the
time from some initial conditions.  Interpretability tooling that could tell *which*
initial conditions and why would be useful here in its own right.

## The interpretability connection, concretely

What makes this testbed attractive compared with an LLM is that every latent the network
might represent has ground truth, and there is a known optimal algorithm to compare with:

1. **Ground-truth latents.**  With randomised physics, the hidden parameters (masses,
   lengths, damping) are known per episode.  The question "is the posterior over the
   parameters linearly decodable from the residual stream, and does it sharpen with
   context length" is the control analogue of Othello-GPT's linear board state
   (https://arxiv.org/abs/2309.00941) and of belief-state geometry, where transformers
   trained on HMM outputs carry the Bayesian belief state linearly in the residual stream
   (https://arxiv.org/abs/2405.15943).  There is a recent LLM result that residual
   directions correlated with *energy* appear spontaneously when forecasting toy physics
   in context (https://arxiv.org/html/2508.12448); the same probe is cheap here, with
   exact energy from `physics.energy`.
2. **A known optimal algorithm.**  The Kalman filter + LQR is the optimal solution of the
   linear-Gaussian version of the task, and a transformer can implement it exactly
   (Goel & Bartlett).  One can therefore ask whether the trained network's attention
   heads implement a filter (do they attend to a fixed window of past states with
   weights resembling Kalman gains?) and whether its output head is an LQR-like readout
   of the estimated state, and localise where it deviates.  This is a sharper question
   than any available for language models, where the "right algorithm" is unknown.
3. **In-context learning with a physical meaning.**  Randomised dynamics turn the
   controller into a meta-learner; whether it learns a Bayesian, task-vector-like
   representation in distribution and something else out of distribution
   (https://arxiv.org/abs/2605.03780) can be tested by sampling parameters outside the
   training range and probing.  Vafa et al. warn that sequence models can fit
   trajectories with heuristics rather than Newtonian world models
   (https://arxiv.org/abs/2507.06952); a small control transformer is a good place to
   check that claim mechanistically.
4. **Small enough for full circuit analysis.**  The existing small-RL interpretability
   work (Decision Transformer Interpretability, 1 layer / 2 heads,
   https://www.alignmentforum.org/posts/bBuBDJBYHt39Q5zZy/decision-transformer-interpretability;
   the maze "cheese vector",
   https://www.alignmentforum.org/posts/cAC4AXiNC5ig6jQnc/understanding-and-controlling-a-maze-solving-policy-network;
   emergent planning in a Sokoban agent, https://arxiv.org/abs/2504.01871) all rely on
   models this size.  Sparse autoencoders on control policies are essentially unexplored
   (one student project on a Pac-Man DQN).

Gaps the literature search confirmed: nobody has trained a transformer policy on a
randomised N-link pendulum and probed it for the hidden parameters, nobody has compared a
trained PPO pendulum policy against LQR mechanistically, and SAEs on controllers are
untouched.  All three are feasible with what is in this repo.

## If you want to try it: a concrete plan

1. **Make memory necessary.**  Add `randomize=True` to the env: per episode sample
   link masses, lengths, drag diameter and friction from ranges (say ±40%), and do not
   put them in the observation.  Optionally add `observe_velocities=False`.
2. **Baselines first.**  MLP on the Markov state (will fail on randomised physics: it
   cannot identify the system), then `RecurrentPPO` with an LSTM from sb3-contrib
   (https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html), then a
   1-D conv over the last 32 steps (the RMA design, https://arxiv.org/abs/2107.04034).
   POPGym says the LSTM may win; that is fine, the transformer is for interpretability.
3. **Transformer policy.**  A causal transformer over the last 32–64 (state, action)
   pairs, 2 layers, 4 heads, d_model 64, built with TransformerLens's
   `HookedTransformerConfig` (Bloom's DT work is the precedent for a continuous input
   embedding) so hooks, activation patching and attention analysis come for free; or wrap
   any PyTorch policy with nnsight (https://github.com/ndif-team/nnsight).  Train with
   PPO via a custom SB3 policy, or offline (Decision-Transformer style) on logged
   trajectories from the existing agents.
4. **Probes and interventions.**  Linear probes for the hidden parameters, for energy,
   and for the Kalman/LQR quantities computed from `physics.py`; attention-pattern
   analysis against the Kalman-gain window; activation patching of the inferred
   parameters (swap the residual-stream direction for "heavy top link" into a light-link
   episode and watch the force change); SAEs on the residual stream via SAELens with a
   custom activation store.
5. **Compare with the MLP story.**  `probe_linear.py` already shows the MLPs act along the
   LQR direction with saturation; the same measurement on the transformer, per context
   length, shows whether the learned filter improves the gain estimate as evidence
   accumulates.

## What is established versus speculative

Established: transformers can implement Kalman filtering and LQG; history-conditioned
policies adapt to hidden dynamics and MLPs on the Markov state cannot; MLPs match
Decision Transformers on standard offline benchmarks; small sequence models linearly
represent latent generative state (Othello, belief states); tiny RL policies are tractable
for circuit-level analysis.

Speculative: that a transformer beats an LSTM on a randomised 1–3-link stick (no direct
study); that physical parameters are linearly decodable from a control transformer's
residual stream (plausible by analogy, untested); that SAE features on a 64-unit
controller mean anything (unknown).
