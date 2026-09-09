#!/usr/bin/env bash
# Train the whole ladder: 1 link, then 2, then 3.  Each run stops early once solved.
set -euo pipefail
cd "$(dirname "$0")"
for n in 1 2 3; do
  python train.py --links "$n" --algo ppo "$@" 2>&1 | tee "runs/ppo_${n}links.log"
  python evaluate.py --links "$n" --algo ppo --export "runs/ppo_${n}links/trajectory.json"
done
