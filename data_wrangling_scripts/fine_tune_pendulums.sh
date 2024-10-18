#!/bin/bash
trap 'kill $(jobs -p)' EXIT

for file in trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:*/seeds/*/epochs/5; do
    python -m envs.Pendulum.train_pendulum -p "$file" --gravity 4 --epochs 5 --seed 1 -a_n 0.01 -a &
    sleep 1
    python -m envs.Pendulum.train_pendulum -p "$file" --gravity 4 --epochs 5 --seed 1 -a_n 0.01 &
    sleep 1
done

wait
