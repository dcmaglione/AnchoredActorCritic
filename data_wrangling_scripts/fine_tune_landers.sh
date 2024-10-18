trap 'kill $(jobs -p)' EXIT
for file in trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,w:True,x:EEUBTEYC3HCGRJC/seeds/*/epochs/49; do
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 -a --epochs 14 &
    sleep 1
    python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 --epochs 14 &
    sleep 1
done
wait