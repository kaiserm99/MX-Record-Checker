#!/usr/bin/env bash
cd "$(dirname "$0")"
rm -rf runs/ppo_1links_swingup runs/ppo_1links_swingup_shake runs/ppo_1links_shake
./run_swingup.sh 1
./run_shake.sh 1
