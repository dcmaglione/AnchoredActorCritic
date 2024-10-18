#!/bin/bash
trap 'kill $(jobs -p)' EXIT

# Original
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:*/seeds/*/epochs/5 --gravity 13 --store_results results/pendulum/original/Source.pkl &
sleep 1
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.1,e:6,g:13,l:0.0001,s_s:1000,x:*/seeds/*/epochs/5 --gravity 4 --store_results results/pendulum/original/Target.pkl &
sleep 1

# Naive
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.01,e:6,g:4,l:0.0001,p:*,s_s:1000,x:*/seeds/1/epochs/5 --gravity 13 --store_results results/pendulum/naive/Source.pkl &
sleep 1
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a_n:0.01,e:6,g:4,l:0.0001,p:*,s_s:1000,x:*/seeds/1/epochs/5 --gravity 4 --store_results results/pendulum/naive/Target.pkl &
sleep 1

# Anchored
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a:True,a_n:0.01,e:6,g:4,l:0.0001,p:*,s_s:1000,x:*/seeds/1/epochs/5 --gravity 13 --store_results results/pendulum/anchored/Source.pkl &
sleep 1
python -m envs.Pendulum.test_pendulum -r trained/Pendulum-custom/a:True,a_n:0.01,e:6,g:4,l:0.0001,p:*,s_s:1000,x:*/seeds/1/epochs/5 --gravity 4 --store_results results/pendulum/anchored/Target.pkl &

wait
