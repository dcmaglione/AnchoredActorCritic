#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directory and delete existing file if it exists
mkdir -p results/lander/original
rm -f results/lander/original/paths.txt

train_lander() {
    local seed=$1
    PYTHONUNBUFFERED=1 python -m envs.LunarLander.train_lander --seed "$seed" --replay_save --epochs 30 |
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> results/lander/original/paths.txt)
}

for i in $(seq 1 12); do
    train_lander $i &
    sleep 1
done

wait
