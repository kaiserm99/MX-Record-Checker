#!/usr/bin/env bash
# Remaining work.  The 2-link swing-up continues from its stage-1 model (learned the swing,
# not the hold) with gentler, more frequent upright starts.
cd "$(dirname "$0")"
python train.py --links 2 --task swingup --init-from runs/ppo_2links_swingup_stage1 2>&1 | tee runs/ppo_2links_swingup.log
python evaluate.py --links 2 --task swingup --export runs/ppo_2links_swingup/trajectory.json
python train.py --links 2 --task swingup --phase shake 2>&1 | tee runs/ppo_2links_swingup_shake.log
python evaluate.py --links 2 --task swingup --phase shake --export runs/ppo_2links_swingup_shake/trajectory.json
./run_swingup.sh 3
./run_swingup.sh 1
./run_shake.sh 1
