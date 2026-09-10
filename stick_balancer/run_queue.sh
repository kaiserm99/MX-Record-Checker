#!/usr/bin/env bash
cd "$(dirname "$0")"
# 2 links: swing stage from the saved hold weights, then pushes
python train.py --links 2 --algo ppo --task swingup --env-override upright_reset_prob=0.5 \
    --init-from runs/ppo_2links_swingup_hold 2>&1 | tee runs/ppo_2links_swingup.log
python evaluate.py --links 2 --algo ppo --task swingup --export runs/ppo_2links_swingup/trajectory.json
python train.py --links 2 --algo ppo --task swingup --phase shake 2>&1 | tee runs/ppo_2links_swingup_shake.log
python evaluate.py --links 2 --algo ppo --task swingup --phase shake --export runs/ppo_2links_swingup_shake/trajectory.json
./run_swingup.sh 3
./run_swingup.sh 1
./run_shake.sh 1
