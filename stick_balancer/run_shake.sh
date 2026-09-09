#!/usr/bin/env bash
# Push phase only, for agents that are already balanced (runs/ppo_<n>links exists).
set -uo pipefail
cd "$(dirname "$0")"
for n in "$@"; do
  python train.py --links "$n" --algo ppo --phase shake 2>&1 | tee "runs/ppo_${n}links_shake.log"
  python evaluate.py --links "$n" --algo ppo --phase shake --export "runs/ppo_${n}links_shake/trajectory.json"
done
