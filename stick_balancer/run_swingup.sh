#!/usr/bin/env bash
# Swing-up ladder: for each link count, learn to swing up and balance, then add pushes.
set -uo pipefail
cd "$(dirname "$0")"
for n in "$@"; do
  python train.py --links "$n" --algo ppo --task swingup 2>&1 | tee "runs/ppo_${n}links_swingup.log"
  python evaluate.py --links "$n" --algo ppo --task swingup --export "runs/ppo_${n}links_swingup/trajectory.json"
  python train.py --links "$n" --algo ppo --task swingup --phase shake 2>&1 | tee "runs/ppo_${n}links_swingup_shake.log"
  python evaluate.py --links "$n" --algo ppo --task swingup --phase shake --export "runs/ppo_${n}links_swingup_shake/trajectory.json"
done
