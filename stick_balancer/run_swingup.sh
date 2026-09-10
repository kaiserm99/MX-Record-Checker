#!/usr/bin/env bash
# Swing-up ladder, per link count: (1) hold stage, every episode starts nearly upright;
# (2) swing stage, continuing from the hold weights with half the episodes hanging;
# (3) pushes on top of the swing-up agent.
set -uo pipefail
cd "$(dirname "$0")"
for n in "$@"; do
  python train.py --links "$n" --algo ppo --task swingup --env-override upright_reset_prob=1.0 \
      --out "runs/ppo_${n}links_swingup_hold" 2>&1 | tee "runs/ppo_${n}links_swingup_hold.log"
  python train.py --links "$n" --algo ppo --task swingup --env-override upright_reset_prob=0.5 \
      --init-from "runs/ppo_${n}links_swingup_hold" 2>&1 | tee "runs/ppo_${n}links_swingup.log"
  python evaluate.py --links "$n" --algo ppo --task swingup --export "runs/ppo_${n}links_swingup/trajectory.json"
  python train.py --links "$n" --algo ppo --task swingup --phase shake 2>&1 | tee "runs/ppo_${n}links_swingup_shake.log"
  python evaluate.py --links "$n" --algo ppo --task swingup --phase shake --export "runs/ppo_${n}links_swingup_shake/trajectory.json"
done
