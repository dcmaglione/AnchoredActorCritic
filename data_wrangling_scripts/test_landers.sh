#!/bin/bash

trap 'kill $(jobs -p)' EXIT

trained_original="trained/lander-custom/a_n:0.01,e:50,l:0.001,s_s:10000,x:YQJBZMB5V5LRVHI/seeds/*/epochs/49"

# Define arrays for trained_naive and trained_anchored
trained_naive=(
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:RVWBW5Y,s_s:15000,x:WT3XD2POHV6ZLJW/seeds/1/epochs/39"
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:PEJYDXC,s_s:15000,x:TZDWP4LRAMP6LIO/seeds/1/epochs/39"
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:EAUU6QD,s_s:15000,x:JAJE2X3VQB4HCYA/seeds/1/epochs/39"
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:FSO3LCW,s_s:15000,x:QMDJZJ3VTMGRRE5/seeds/1/epochs/39"
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:LGFJZP3,s_s:15000,x:WWHBSZL3HKWRXTJ/seeds/1/epochs/39"
    "trained/lander-custom/a_n:0.01,e:40,l:0.001,p:TF6CGLY,s_s:15000,x:UQC2GFI3WULBCCL/seeds/1/epochs/39"
)

trained_anchored=(
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:PEJYDXC,s_s:15000,x:TZDWP4LRAMP6LIO/seeds/1/epochs/39"
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:RVWBW5Y,s_s:15000,x:WT3XD2POHV6ZLJW/seeds/1/epochs/39"
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:FSO3LCW,s_s:15000,x:QMDJZJ3VTMGRRE5/seeds/1/epochs/39"
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:LGFJZP3,s_s:15000,x:WWHBSZL3HKWRXTJ/seeds/1/epochs/39"
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:TF6CGLY,s_s:15000,x:UQC2GFI3WULBCCL/seeds/1/epochs/39"
    "trained/lander-custom/a:True,a_n:0.01,e:40,l:0.001,p:EAUU6QD,s_s:15000,x:JAJE2X3VQB4HCYA/seeds/1/epochs/39"
)



# Echo expanded paths
echo "Original paths:"
echo $trained_original

echo -e "\nNaive paths:"
printf '%s\n' "${trained_naive[@]}"

echo -e "\nAnchored paths:"
printf '%s\n' "${trained_anchored[@]}"

# Function to run tests
run_tests() {
    local config_type=$1
    shift
    local paths=("$@")
    python -m envs.LunarLander.test_lander "${paths[@]}" --store_results "results/lander/${config_type}/Source.pkl" --initial-random 1500.0 &
    sleep 1
    python -m envs.LunarLander.test_lander "${paths[@]}" --store_results "results/lander/${config_type}/Target.pkl" --initial-random 500.0 &
    sleep 1
}

# # Run tests for original configurations
run_tests "original" $trained_original

# Run tests for naive configurations
run_tests "naive" "${trained_naive[@]}"

# Run tests for anchored configurations
run_tests "anchored" "${trained_anchored[@]}"

# Wait for all background jobs to finish
wait
