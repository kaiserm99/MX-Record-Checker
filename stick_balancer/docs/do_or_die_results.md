# Do-or-die results

Test windows: 1800, context K = 32.

## A. Imitation error versus context length (test MSE, lower is better)

| context steps | state MLP | transformer | oracle MLP |
|---|---|---|---|
| 1 | 0.01156 | 0.01309 | 0.00280 |
| 2 | 0.00835 | 0.00495 | 0.00252 |
| 4 | 0.00499 | 0.00300 | 0.00182 |
| 8 | 0.00229 | 0.00180 | 0.00083 |
| 16 | 0.00231 | 0.00124 | 0.00065 |
| 32 | 0.00170 | 0.00092 | 0.00038 |

With the full context the transformer closes 59% of the gap between the state-only floor and the oracle ceiling.

## B. Linear probes for the hidden parameters (test R², ridge regression)

**after embedding**

| context steps | link_lengths | link_masses | actuator_gain |
|---|---|---|---|
| 1 | -0.00 | -0.00 | -0.00 |
| 2 | -0.00 | -0.00 | -0.00 |
| 4 | -0.00 | -0.01 | -0.00 |
| 8 | -0.00 | -0.00 | -0.00 |
| 16 | -0.00 | -0.00 | -0.01 |
| 32 | -0.00 | -0.00 | -0.00 |

**after block 1**

| context steps | link_lengths | link_masses | actuator_gain |
|---|---|---|---|
| 1 | 0.13 | -0.00 | 0.36 |
| 2 | 0.20 | -0.00 | 0.59 |
| 4 | 0.28 | -0.01 | 0.59 |
| 8 | 0.36 | -0.00 | 0.60 |
| 16 | 0.41 | 0.00 | 0.63 |
| 32 | 0.47 | -0.00 | 0.67 |

**after block 2**

| context steps | link_lengths | link_masses | actuator_gain |
|---|---|---|---|
| 1 | 0.12 | 0.00 | 0.35 |
| 2 | 0.15 | -0.01 | 0.56 |
| 4 | 0.18 | 0.00 | 0.59 |
| 8 | 0.24 | 0.00 | 0.62 |
| 16 | 0.29 | 0.00 | 0.65 |
| 32 | 0.31 | 0.01 | 0.74 |

Control: a linear probe on the raw 32-step input window (no network) gives R² = link_lengths -0.02, link_masses -0.01, actuator_gain -0.02.

## C. Closed loop on fresh hidden physics (transformer as the controller)

- survived all 200 steps: 87/100 episodes
- correlation between the network's fitted action gain and the optimal one across episodes (log-log): r = 0.76
- mean cosine between the network's gain vector and the LQR direction: 0.929

## D. Attention weight by lag (last position, averaged over test windows)

| layer | head | lag 0 | lag 1 | lag 2 | lag 4 | lag 8 | lag 16 | lag 31 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 1 | 1 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 1 | 2 | 0.02 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 1 | 3 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |
| 2 | 0 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.05 |
| 2 | 1 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.03 | 0.03 |
| 2 | 2 | 0.05 | 0.04 | 0.04 | 0.04 | 0.03 | 0.03 | 0.02 |
| 2 | 3 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 |

## E. Causal patching along the probe direction for the actuator gain

Prediction from physics: if the network believes the gain is c times larger, its action should scale by 1/c.  We add (log c) * w/|w|² along the probe direction w at every position and measure the actual change of the last-position action.

| layer | c | corr(actual Δa, predicted Δa) | slope actual/predicted | mean |actual| / mean |predicted| |
|---|---|---|---|---|
| block 1 | 1.50 | -0.08 | -0.00 | 0.03 |
| block 1 | 0.67 | -0.07 | -0.00 | 0.02 |
| block 2 | 1.50 | 0.87 | 0.00 | 0.00 |
| block 2 | 0.67 | 0.80 | 0.00 | 0.00 |
| block 2, random direction (control) | 1.50 | 0.09 | 0.00 | 0.07 |

## F. Necessity: project the actuator-gain probe direction out of the residual stream

If the network *uses* that direction, removing it should hurt imitation; removing a random direction should not.

| layer | direction removed | test MSE (last position) |
|---|---|---|
| – | nothing | 0.00092 |
| block 1 | probe direction | 0.00093 |
| block 1 | random direction | 0.00094 |
| block 2 | probe direction | 0.00092 |
| block 2 | random direction | 0.00125 |

## G. Dose-response: output change versus patch size along the block-1 probe direction

| multiple of the 'gain x 1.5' step | mean Δ action | corr with physics prediction |
|---|---|---|
| 1 | +0.0006 | -0.08 |
| 4 | +0.0025 | -0.10 |
| 16 | +0.0103 | -0.17 |
| 64 | +0.0437 | -0.34 |

(For scale: the physics prediction has mean Δ action -0.0001.)
