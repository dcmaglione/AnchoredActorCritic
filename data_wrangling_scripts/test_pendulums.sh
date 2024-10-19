#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Read trained models from files
mapfile -t trained_original < results/pendulum/original/paths.txt
mapfile -t trained_naive < results/pendulum/naive/paths.txt
mapfile -t trained_anchored < results/pendulum/anchored/paths.txt

# Echo expanded paths
echo "Original paths:"
printf '%s\n' "${trained_original[@]}"

echo -e "\nNaive paths:"
printf '%s\n' "${trained_naive[@]}"

echo -e "\nAnchored paths:"
printf '%s\n' "${trained_anchored[@]}"

# Function to run tests
run_tests() {
    local config_type=$1
    shift
    local paths=("$@")
    echo "Testing ${config_type} configuration on source environment:"
    PYTHONUNBUFFERED=1 python -m envs.Pendulum.test_pendulum "${paths[@]}" --store_results "results/pendulum/${config_type}/Source.pkl" --gravity 13 &
    sleep 1
    echo "Testing ${config_type} configuration on target environment:"
    PYTHONUNBUFFERED=1 python -m envs.Pendulum.test_pendulum "${paths[@]}" --store_results "results/pendulum/${config_type}/Target.pkl" --gravity 4 &
    sleep 1
}

# Run tests for original configurations
run_tests "original" "${trained_original[@]}"

# Run tests for naive configurations
run_tests "naive" "${trained_naive[@]}"

# Run tests for anchored configurations
run_tests "anchored" "${trained_anchored[@]}"

# Wait for all background jobs to finish
wait
