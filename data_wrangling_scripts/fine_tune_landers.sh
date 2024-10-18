trap 'kill $(jobs -p)' EXIT
for file in trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,x:YQJBZMB5V5LRVHI/seeds/*/epochs/49; do
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 -a --initial-random 100.0 --epochs 14 &
    sleep 1
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 --initial-random 100.0 --epochs 14 &
    sleep 1
done
wait