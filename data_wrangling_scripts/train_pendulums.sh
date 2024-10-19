#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directory and delete existing file if it exists
mkdir -p results/pendulum/original
rm -f results/pendulum/original/paths.txt

train_pendulum() {
    local seed=$1
    PYTHONUNBUFFERED=1 python -m envs.Pendulum.train_pendulum --replay_save --seed "$seed" --epochs 7 --gravity 13 |
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> results/pendulum/original/paths.txt)
}

for i in $(seq 1 12); do
    train_pendulum $i &
    sleep 1
done

wait
