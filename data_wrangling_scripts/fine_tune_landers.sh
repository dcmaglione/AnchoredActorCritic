trap 'kill $(jobs -p)' EXIT
for file in "trained/lander-custom/a_n:0.01,e:51,l:(0.001,1e-05),s_s:10000,x:WTRFIURF5FIRZUQ/seeds/"*/epochs/50; do
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 -a --initial-random 500.0 --epochs 10 &
    sleep 1
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 --initial-random 500.0 --epochs 10 &
    sleep 1
done
wait