"""
Step 3: the actual do-or-die measurements on the trained transformer.

  A. Imitation error versus context length (does more history help, and how fast?).
  B. Linear probes: are the hidden parameters (log length, log pole mass, log actuator gain)
     linearly decodable from the residual stream, per layer and per context length?
  C. Closed loop: run the transformer as the controller on fresh hidden physics; does its
     effective gain track the actuator gain the way the optimal controller's does?
  D. Attention: mean attention weight as a function of lag, per head.
  E. Causal patching: shift the residual stream along the probe direction for the actuator
     gain by an amount corresponding to "gain x c", and compare the change in the output
     action with the physics prediction (the optimal action scales by 1/c).

    python interp/analyze.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from collect import HIDDEN, make  # noqa: E402
from lqr import lqr_gain  # noqa: E402
from models import OracleMLP, StateMLP, TinyTransformer  # noqa: E402
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("interp_train", HERE / "train.py")
_interp_train = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_interp_train)
load_windows = _interp_train.load_windows

OUT = HERE / "results.md"


def ridge(X, y, lam=1e-2):
    """Closed-form ridge regression with intercept; returns (w, b)."""
    Xm, ym = X.mean(0), y.mean()
    Xc, yc = X - Xm, y - ym
    w = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return w, ym - Xm @ w


def r2(y, yhat):
    return 1 - np.var(y - yhat) / np.var(y)


def main():
    torch.set_num_threads(2)
    cfg = json.loads((HERE / "models" / "results.json").read_text())
    K = cfg["context"]
    (X, X_te), (Y, Y_te), (Hp, Hp_te), _ = load_windows(HERE / "data.npz", K)
    tf = TinyTransformer(context=K); tf.load_state_dict(torch.load(HERE / "models" / "transformer.pt")); tf.eval()
    mlp = StateMLP(); mlp.load_state_dict(torch.load(HERE / "models" / "state_mlp.pt")); mlp.eval()
    orc = OracleMLP(3); orc.load_state_dict(torch.load(HERE / "models" / "oracle_mlp.pt")); orc.eval()
    lines = ["# Do-or-die results", "", f"Test windows: {len(X_te)}, context K = {K}.", ""]

    # ---------------- A. error vs context length ----------------
    with torch.no_grad():
        p_tf, resid = tf(X_te, return_residuals=True)
        p_mlp = mlp(X_te)
        p_orc = orc(X_te, Hp_te)
    err = lambda p: ((p - Y_te) ** 2).mean(0).numpy()  # noqa: E731
    e_tf, e_mlp, e_orc = err(p_tf), err(p_mlp), err(p_orc)
    lines += ["## A. Imitation error versus context length (test MSE, lower is better)", "",
              "| context steps | state MLP | transformer | oracle MLP |", "|---|---|---|---|"]
    for pos in (0, 1, 3, 7, 15, 31):
        lines.append(f"| {pos + 1} | {e_mlp[pos]:.5f} | {e_tf[pos]:.5f} | {e_orc[pos]:.5f} |")
    gap_closed = 1 - (e_tf[K - 1] - e_orc[K - 1]) / max(e_mlp[K - 1] - e_orc[K - 1], 1e-9)
    lines += ["", f"With the full context the transformer closes {gap_closed:.0%} of the gap between the "
              "state-only floor and the oracle ceiling.", ""]

    # ---------------- B. probes ----------------
    lines += ["## B. Linear probes for the hidden parameters (test R², ridge regression)", ""]
    H_te = Hp_te.numpy()
    H_tr = Hp.numpy()
    with torch.no_grad():
        _, resid_tr = tf(X, return_residuals=True)
    probe_dirs = {}
    for li, name in enumerate(["after embedding", "after block 1", "after block 2"]):
        lines += [f"**{name}**", "", "| context steps | " + " | ".join(HIDDEN) + " |", "|---|" + "---|" * len(HIDDEN)]
        for pos in (0, 1, 3, 7, 15, 31):
            cells = []
            for j, h in enumerate(HIDDEN):
                w, b = ridge(resid_tr[li][:, pos].numpy(), H_tr[:, j])
                cells.append(f"{r2(H_te[:, j], resid[li][:, pos].numpy() @ w + b):.2f}")
                if pos == K - 1:
                    probe_dirs[(li, h)] = (w, b)
            lines.append(f"| {pos + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    # sanity: the same probe on the raw input window (no network) at full context
    raw = X_te.reshape(len(X_te), -1).numpy(); raw_tr = X.reshape(len(X), -1).numpy()
    cells = []
    for j in range(len(HIDDEN)):
        w, b = ridge(raw_tr, H_tr[:, j], lam=1.0)
        cells.append(f"{r2(H_te[:, j], raw @ w + b):.2f}")
    lines += ["Control: a linear probe on the raw 32-step input window (no network) gives R² = "
              + ", ".join(f"{h} {c}" for h, c in zip(HIDDEN, cells)) + ".", ""]

    # ---------------- C. closed loop ----------------
    env = make()
    norm = cfg["norm"]; mean, std = np.array(norm["mean"], np.float32), np.array(norm["std"], np.float32)
    survived, gains, gain_true, cos_lqr = 0, [], [], []
    n_eps, steps = 100, 200
    for ep in range(n_eps):
        obs, _ = env.reset(seed=10_000 + ep)
        K_lqr = lqr_gain(env)
        hist, states, acts = [], [], []
        a_prev = 0.0
        for t in range(steps):
            hist.append(np.concatenate([obs, [a_prev]]).astype(np.float32))
            window = np.array(hist[-K:])
            tok = torch.tensor((window - mean) / std)[None]
            with torch.no_grad():
                a = float(tf(tok)[0, -1].clamp(-1, 1))
            states.append(obs.copy()); acts.append(a)
            obs, _, terminated, _, _ = env.step([a])
            a_prev = a
            if terminated:
                break
        survived += t == steps - 1
        S, A = np.array(states)[K:], np.array(acts)[K:]   # after the context has filled
        if len(S) > 20 and np.abs(A).max() > 1e-3:
            coef, *_ = np.linalg.lstsq(np.hstack([S, np.ones((len(S), 1))]), A, rcond=None)
            k_net = -coef[:4]
            k_opt = K_lqr / (env.max_force * env.actuator_gain)
            gains.append(np.linalg.norm(k_net)); gain_true.append(np.linalg.norm(k_opt))
            cos_lqr.append(k_net @ k_opt / (np.linalg.norm(k_net) * np.linalg.norm(k_opt) + 1e-12))
    rho = np.corrcoef(np.log(gains), np.log(gain_true))[0, 1]
    lines += ["## C. Closed loop on fresh hidden physics (transformer as the controller)", "",
              f"- survived all {steps} steps: {survived}/{n_eps} episodes",
              f"- correlation between the network's fitted action gain and the optimal one across episodes (log-log): r = {rho:.2f}",
              f"- mean cosine between the network's gain vector and the LQR direction: {np.mean(cos_lqr):.3f}", ""]

    # ---------------- D. attention ----------------
    with torch.no_grad():
        tf(X_te[:256])
    lines += ["## D. Attention weight by lag (last position, averaged over test windows)", "",
              "| layer | head | lag 0 | lag 1 | lag 2 | lag 4 | lag 8 | lag 16 | lag 31 |", "|---|---|---|---|---|---|---|---|---|"]
    for li, blk in enumerate(tf.blocks, start=1):
        w = blk.last_attn[:, :, K - 1, :].mean(0)  # (heads, K) attention from last token
        for h in range(w.shape[0]):
            row = [f"{w[h, K - 1 - lag]:.2f}" for lag in (0, 1, 2, 4, 8, 16, 31)]
            lines.append(f"| {li} | {h} | " + " | ".join(row) + " |")
    lines.append("")

    # ---------------- E. patching ----------------
    lines += ["## E. Causal patching along the probe direction for the actuator gain", "",
              "Prediction from physics: if the network believes the gain is c times larger, its action "
              "should scale by 1/c.  We add (log c) * w/|w|² along the probe direction w at every "
              "position and measure the actual change of the last-position action.", "",
              "| layer | c | corr(actual Δa, predicted Δa) | slope actual/predicted | mean |actual| / mean |predicted| |",
              "|---|---|---|---|---|"]
    base = p_tf[:, K - 1].numpy()
    for li, name in ((1, "block 1"), (2, "block 2")):
        w, _ = probe_dirs[(li, "actuator_gain")]
        for c in (1.5, 1 / 1.5):
            delta = torch.tensor(np.log(c) * w / (w @ w), dtype=torch.float32)
            with torch.no_grad():
                patched = tf.forward_patched(X_te, li, delta)[:, K - 1].numpy()
            actual = patched - base
            predicted = base / c - base
            corr = np.corrcoef(actual, predicted)[0, 1]
            slope = (actual @ predicted) / (predicted @ predicted)
            ratio = np.abs(actual).mean() / np.abs(predicted).mean()
            lines.append(f"| {name} | {c:.2f} | {corr:.2f} | {slope:.2f} | {ratio:.2f} |")
    # control: a random direction of the same norm
    rng = np.random.default_rng(0)
    w, _ = probe_dirs[(2, "actuator_gain")]
    rnd = rng.standard_normal(len(w)); rnd *= np.linalg.norm(np.log(1.5) * w / (w @ w)) / np.linalg.norm(rnd)
    with torch.no_grad():
        patched = tf.forward_patched(X_te, 2, torch.tensor(rnd, dtype=torch.float32))[:, K - 1].numpy()
    actual = patched - base; predicted = base / 1.5 - base
    lines.append(f"| block 2, random direction (control) | 1.50 | {np.corrcoef(actual, predicted)[0, 1]:.2f} | "
                 f"{(actual @ predicted) / (predicted @ predicted):.2f} | {np.abs(actual).mean() / np.abs(predicted).mean():.2f} |")
    lines.append("")

    # ---------------- F. necessity: remove the probe direction ----------------
    lines += ["## F. Necessity: project the actuator-gain probe direction out of the residual stream", "",
              "If the network *uses* that direction, removing it should hurt imitation; removing a random "
              "direction should not.", "", "| layer | direction removed | test MSE (last position) |", "|---|---|---|"]
    lines.append(f"| – | nothing | {e_tf[K - 1]:.5f} |")
    for li, name in ((1, "block 1"), (2, "block 2")):
        w, _ = probe_dirs[(li, "actuator_gain")]
        u = torch.tensor(w / np.linalg.norm(w), dtype=torch.float32)
        for label, direction in (("probe direction", u), ("random direction", torch.tensor(rng.standard_normal(len(w)) / np.sqrt(len(w)), dtype=torch.float32))):
            direction = direction / direction.norm()
            with torch.no_grad():
                Kk = X_te.shape[1]
                x = tf.embed(X_te) + tf.pos[:, :Kk]
                for i, blk in enumerate(tf.blocks, start=1):
                    x = blk(x, tf.mask[:Kk, :Kk])
                    if i == li:
                        x = x - (x @ direction)[..., None] * direction   # remove that component
                pred = tf.head(tf.ln_f(x)).squeeze(-1)[:, K - 1]
            mse = ((pred - Y_te[:, K - 1]) ** 2).mean().item()
            lines.append(f"| {name} | {label} | {mse:.5f} |")
    lines.append("")

    # ---------------- G. dose-response of the patch ----------------
    lines += ["## G. Dose-response: output change versus patch size along the block-1 probe direction", "",
              "| multiple of the 'gain x 1.5' step | mean Δ action | corr with physics prediction |", "|---|---|---|"]
    w, _ = probe_dirs[(1, "actuator_gain")]
    unit = torch.tensor(np.log(1.5) * w / (w @ w), dtype=torch.float32)
    predicted = base / 1.5 - base
    for m in (1, 4, 16, 64):
        with torch.no_grad():
            patched = tf.forward_patched(X_te, 1, m * unit)[:, K - 1].numpy()
        actual = patched - base
        lines.append(f"| {m} | {actual.mean():+.4f} | {np.corrcoef(actual, predicted)[0, 1]:.2f} |")
    lines += ["", f"(For scale: the physics prediction has mean Δ action {predicted.mean():+.4f}.)", ""]

    OUT.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
