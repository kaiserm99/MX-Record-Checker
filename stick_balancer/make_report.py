"""
Bundle learning curves and trajectories from runs/ into one self-contained HTML page
(report.html) with a canvas animation of every trained agent.

    python make_report.py            # reads runs/ppo_*links/, writes report.html
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

RUNS = Path("runs")
TEMPLATE = Path("report_template.html")


def learning_curve(run_dir: Path) -> list[tuple[int, float]]:
    """(timesteps, mean episode reward) pairs from SB3's progress.csv."""
    path = run_dir / "progress.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for row in csv.DictReader(f):
            t, r = row.get("time/total_timesteps"), row.get("rollout/ep_rew_mean")
            if t and r:
                rows.append((int(float(t)), round(float(r), 2)))
    return rows


def eval_curve(run_dir: Path) -> list[tuple[int, float]]:
    """(timesteps, mean deterministic eval return) pairs from EvalCallback's npz."""
    path = run_dir / "evaluations.npz"
    if not path.exists():
        return []
    import numpy as np

    data = np.load(path)
    return [(int(t), round(float(r.mean()), 2)) for t, r in zip(data["timesteps"], data["results"])]


def main() -> None:
    runs = []
    for run_dir in sorted(RUNS.glob("*links")):
        traj = run_dir / "trajectory.json"
        runs.append({
            "name": run_dir.name,
            "config": json.loads((run_dir / "config.json").read_text()) if (run_dir / "config.json").exists() else {},
            "train_curve": learning_curve(run_dir),
            "eval_curve": eval_curve(run_dir),
            "trajectory": json.loads(traj.read_text()) if traj.exists() else None,
        })
    sources = json.loads(Path("sources.json").read_text()) if Path("sources.json").exists() else []
    html = (
        TEMPLATE.read_text()
        .replace("/*__RUNS__*/[]", json.dumps(runs))
        .replace("/*__SOURCES__*/[]", json.dumps(sources))
    )
    Path("report.html").write_text(html)
    print("wrote report.html with", len(runs), "runs")


if __name__ == "__main__":
    main()
