#!/usr/bin/env bash
# 1-link swing-up now; 2 and 3 links once the push-phase chain has finished.
cd "$(dirname "$0")"
./run_swingup.sh 1
while pgrep -f "^bash ./run_after_balance.sh" > /dev/null; do sleep 10; done
./run_swingup.sh 2 3
