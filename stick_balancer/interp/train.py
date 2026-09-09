"""
Step 2: train the three models on the collected dataset by imitation (MSE to the expert's
clean action at every position of a K-step window).

    python interp/train.py --data interp/data.npz --out interp/models
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import OracleMLP, StateMLP, TinyTransformer  # noqa: E402


def load_windows(path: Path, context: int, seed: int = 0):
    """Cut every episode into non-overlapping K-step windows.  Returns train/test splits
    (split by episode so the test physics draws are unseen) as float32 tensors:
    tokens (N, K, 5), targets (N, K), hidden (N, 3) in log space."""
    d = np.load(path)
    S, Aex, Aexp, H, L = d["states"], d["action_exec"], d["action_expert"], d["hidden"], d["lengths"]
    tokens, targets, hidden = [], [], []
    # token t = (state_t, executed action at t-1): the action that *produced* state_t.
    # Never the action at t itself, which is the label plus noise (that would be a leak).
    Aprev = np.concatenate([np.zeros((Aex.shape[0], 1), Aex.dtype), Aex[:, :-1]], axis=1)
    for ep in range(len(L)):
        n = L[ep]
        for start in range(0, n - context + 1, context):
            sl = slice(start, start + context)
            tokens.append(np.concatenate([S[ep, sl], Aprev[ep, sl, None]], axis=-1))
            targets.append(Aexp[ep, sl])
            hidden.append(np.log(H[ep]))
    tokens, targets, hidden = map(np.asarray, (tokens, targets, hidden))
    # normalise token features with training statistics
    rng = np.random.default_rng(seed)
    episodes_of_window = np.repeat(np.arange(len(L)), [max(0, (L[ep] - context) // context + 1) for ep in range(len(L))])
    test_eps = set(rng.choice(len(L), size=len(L) // 5, replace=False).tolist())
    is_test = np.array([e in test_eps for e in episodes_of_window])
    mean = tokens[~is_test].reshape(-1, 5).mean(0)
    std = tokens[~is_test].reshape(-1, 5).std(0) + 1e-6
    tokens = (tokens - mean) / std
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    split = lambda a, m: (t(a[~m]), t(a[m]))  # noqa: E731
    return split(tokens, is_test), split(targets, is_test), split(hidden, is_test), dict(mean=mean.tolist(), std=std.tolist())


def fit(model, X, Y, Hp, X_te, Y_te, Hp_te, epochs, lr, batch, oracle=False, log=print):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            pred = model(X[idx], Hp[idx]) if oracle else model(X[idx])
            loss = nn.functional.mse_loss(pred, Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
        sched.step()
        if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                pred = model(X_te, Hp_te) if oracle else model(X_te)
                te = nn.functional.mse_loss(pred, Y_te).item()
            log(f"  epoch {ep:3d}  train mse {tot / n:.5f}  test mse {te:.5f}")
    return te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data.npz")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "models")
    ap.add_argument("--context", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=True)

    (X, X_te), (Y, Y_te), (Hp, Hp_te), norm = load_windows(args.data, args.context, args.seed)
    print(f"windows: train {len(X)}, test {len(X_te)}; K = {args.context}")
    var = Y_te.var().item()
    results = {"target_variance": var}

    for name, model, oracle, epochs in [
        ("state_mlp", StateMLP(), False, args.epochs),
        ("oracle_mlp", OracleMLP(Hp.shape[1]), True, args.epochs),
        ("transformer", TinyTransformer(context=args.context), False, args.epochs),
    ]:
        print(f"== {name} ({sum(p.numel() for p in model.parameters()):,} parameters)")
        te = fit(model, X, Y, Hp, X_te, Y_te, Hp_te, epochs, lr=1e-3, batch=256, oracle=oracle)
        results[name] = {"test_mse": te, "explained_variance": 1 - te / var}
        torch.save(model.state_dict(), args.out / f"{name}.pt")
    (args.out / "results.json").write_text(json.dumps({**results, "norm": norm, "context": args.context}, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "norm"}, indent=2))


if __name__ == "__main__":
    main()
