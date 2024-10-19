#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directory and delete existing file if it exists
mkdir -p results/reacher/original
rm -f results/reacher/original/paths.txt

train_reacher() {
    local seed=$1
    PYTHONUNBUFFERED=1 python -m envs.Reacher.train_reacher --replay_save --seed "$seed" --epochs 70 --distance 0.2 |
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> results/reacher/original/paths.txt)
}

for i in $(seq 1 6); do
    train_reacher $i &
    sleep 1
done

wait
