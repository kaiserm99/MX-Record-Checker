#!/usr/bin/env bash
# Remaining work, in order: 2/3-link pushes, 2/3-link swing-up (+ pushes), 1-link pushes redone.
cd "$(dirname "$0")"
./run_shake.sh 2 3
./run_swingup.sh 2 3
./run_shake.sh 1
