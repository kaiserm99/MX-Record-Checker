#!/usr/bin/env bash
# When the 1-link queue is done: resume the 2-link swing-up curriculum from its stage-4 best
# (0.70 rad level), then its pushes, then the 3-link swing-up ladder.
cd "$(dirname "$0")"
while pgrep -f "^bash ./run_queue_1link.sh" > /dev/null; do sleep 15; done
rm -rf runs/ppo_2links_swingup
python train.py --links 2 --algo ppo --task swingup --tilt-curriculum --env-override upright_reset_tilt=0.7 \
    --init-from runs/ppo_2links_swingup_stage4 2>&1 | tee runs/ppo_2links_swingup.log
python evaluate.py --links 2 --algo ppo --task swingup --export runs/ppo_2links_swingup/trajectory.json
python train.py --links 2 --algo ppo --task swingup --phase shake 2>&1 | tee runs/ppo_2links_swingup_shake.log
python evaluate.py --links 2 --algo ppo --task swingup --phase shake --export runs/ppo_2links_swingup_shake/trajectory.json
./run_swingup.sh 3
