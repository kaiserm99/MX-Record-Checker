#!/usr/bin/env bash
cd "$(dirname "$0")"
# 2 links: resume the reverse curriculum at the level reached (0.70 rad) from the stage-3 best weights
python train.py --links 2 --algo ppo --task swingup --tilt-curriculum --env-override upright_reset_tilt=0.7 \
    --init-from runs/ppo_2links_swingup_stage3 2>&1 | tee runs/ppo_2links_swingup.log
python evaluate.py --links 2 --algo ppo --task swingup --export runs/ppo_2links_swingup/trajectory.json
python train.py --links 2 --algo ppo --task swingup --phase shake 2>&1 | tee runs/ppo_2links_swingup_shake.log
python evaluate.py --links 2 --algo ppo --task swingup --phase shake --export runs/ppo_2links_swingup_shake/trajectory.json
./run_swingup.sh 3
