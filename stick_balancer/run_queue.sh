#!/usr/bin/env bash
# Everything still to train, in order, with the push curriculum:
#   1-link swing-up + pushes, 2/3-link pushes, 2/3-link swing-up (+ pushes), 1-link pushes redone.
cd "$(dirname "$0")"
python train.py --links 1 --task swingup --phase shake 2>&1 | tee runs/ppo_1links_swingup_shake.log
python evaluate.py --links 1 --task swingup --phase shake --export runs/ppo_1links_swingup_shake/trajectory.json
./run_shake.sh 2 3
./run_swingup.sh 2 3
./run_shake.sh 1
