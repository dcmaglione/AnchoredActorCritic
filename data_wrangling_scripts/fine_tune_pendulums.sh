#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directories and delete existing files if they exist
mkdir -p results/pendulum/naive
mkdir -p results/pendulum/anchored
rm -f results/pendulum/naive/paths.txt
rm -f results/pendulum/anchored/paths.txt

fine_tune_naive() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.Pendulum.train_pendulum -p "$file" --gravity 4 --epochs 5 --seed 1 -a_n 0.01 | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

fine_tune_anchored() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.Pendulum.train_pendulum -a -p "$file" --gravity 4 --epochs 5 --seed 1 -a_n 0.01 | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

while IFS= read -r file; do
    echo "$file"
    # Fine-tune without anchors
    fine_tune_naive "$file" results/pendulum/naive/paths.txt &
    sleep 1

    # Fine-tune with anchors
    fine_tune_anchored "$file" results/pendulum/anchored/paths.txt &
    sleep 1
done < results/pendulum/original/paths.txt

wait
