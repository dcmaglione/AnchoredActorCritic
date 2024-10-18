trap 'kill $(jobs -p)' EXIT
for i in $(seq 1 6); do
    python -m envs.LunarLander.train_lander --seed "$i" --replay_save -w --epochs 50 &
    sleep 1
done
wait