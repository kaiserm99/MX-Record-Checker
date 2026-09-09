#!/usr/bin/env bash
# Wait for the running balance chain to finish, then run the push phase for every link count.
cd "$(dirname "$0")"
while pgrep -f "^bash ./run_all.sh" > /dev/null; do sleep 10; done
./run_shake.sh 1 2 3
