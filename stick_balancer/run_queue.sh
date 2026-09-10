#!/usr/bin/env bash
# Remaining work: swing-up (+ pushes) for 2 and 3 links, the 1-link swing-up redone under the
# final task definition, then the 1-link pushes redone.
cd "$(dirname "$0")"
./run_swingup.sh 2 3
./run_swingup.sh 1
./run_shake.sh 1
