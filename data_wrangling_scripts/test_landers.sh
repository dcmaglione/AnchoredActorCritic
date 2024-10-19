#!/bin/bash

trap 'kill $(jobs -p)' EXIT

trained_original="trained/lander-custom/a_n:0.01,e:51,l:(0.001,1e-05),s_s:10000,x:WTRFIURF5FIRZUQ/seeds/*/epochs/50"

# Define arrays for trained_naive and trained_anchored
trained_naive=(
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:TAXGZDV,s_s:15000,x:7CK3LYQWJOK62WG/seeds/1/epochs/9"
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:LFWQFDG,s_s:15000,x:CWIUTXCBQN5F3HB/seeds/1/epochs/9"
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:NSFKCS2,s_s:15000,x:FWNSCXXW7YVUAXA/seeds/1/epochs/9"
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:WZ3RI4I,s_s:15000,x:OKUZTQOAPAJTVPR/seeds/1/epochs/9"
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:XPJKOT7,s_s:15000,x:7OGNTNLQ2F6T2EY/seeds/1/epochs/9"
    "trained/lander-custom/a_n:0.01,e:10,l:1e-05,p:I4AXL64,s_s:15000,x:7W3XFW6NVWI35ZT/seeds/1/epochs/9"
)

trained_anchored=(
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:TAXGZDV,s_s:15000,x:7CK3LYQWJOK62WG/seeds/1/epochs/9"
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:WZ3RI4I,s_s:15000,x:OKUZTQOAPAJTVPR/seeds/1/epochs/9"
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:I4AXL64,s_s:15000,x:7W3XFW6NVWI35ZT/seeds/1/epochs/9"
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:LFWQFDG,s_s:15000,x:CWIUTXCBQN5F3HB/seeds/1/epochs/9"
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:NSFKCS2,s_s:15000,x:FWNSCXXW7YVUAXA/seeds/1/epochs/9"
    "trained/lander-custom/a:True,a_n:0.01,e:10,l:1e-05,p:XPJKOT7,s_s:15000,x:7OGNTNLQ2F6T2EY/seeds/1/epochs/9"
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
    python -m envs.LunarLander.test_lander "${paths[@]}" --store_results "results/lander/${config_type}/Source.pkl" --initial-random 1000.0 &
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
