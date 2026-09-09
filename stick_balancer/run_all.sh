#!/usr/bin/env bash
# The whole ladder: for 1, 2, 3 links -> balance, balance + pushes, swing-up, swing-up + pushes.
# Each run stops early once solved.
set -euo pipefail
cd "$(dirname "$0")"
for n in 1 2 3; do
  python train.py --links "$n" --algo ppo "$@" 2>&1 | tee "runs/ppo_${n}links.log"
  python evaluate.py --links "$n" --algo ppo --export "runs/ppo_${n}links/trajectory.json"
  python train.py --links "$n" --algo ppo --phase shake "$@" 2>&1 | tee "runs/ppo_${n}links_shake.log"
  python evaluate.py --links "$n" --algo ppo --phase shake --export "runs/ppo_${n}links_shake/trajectory.json"
done
./run_swingup.sh 1 2 3
