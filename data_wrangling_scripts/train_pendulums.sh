#!/bin/bash
trap 'kill $(jobs -p)' EXIT

for i in $(seq 1 6); do
    python -m envs.Pendulum.train_pendulum --replay_save --seed "$i" --epochs 6 --gravity 13 &
    sleep 1
done

wait
