#!/usr/bin/env bash
# Remaining work: swing-up (+ pushes) for 2 and 3 links with the sharpened reward and gSDE,
# then the 1-link swing-up redone under the same reward, then the 1-link pushes redone.
cd "$(dirname "$0")"
./run_swingup.sh 2 3
./run_swingup.sh 1
./run_shake.sh 1
