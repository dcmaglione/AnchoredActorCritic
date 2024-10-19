trap 'kill $(jobs -p)' EXIT
for i in $(seq 1 6); do
    python -m envs.LunarLander.train_lander --seed "$i" --replay_save --initial-random 1000.0 --epochs 51 &
    sleep 1
done
wait