#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directories and delete existing path files if they exist
mkdir -p results/lander/naive
mkdir -p results/lander/anchored
rm -f results/lander/naive/paths.txt
rm -f results/lander/anchored/paths.txt

fine_tune_naive() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 --gravity "-2.0" --epochs 15 | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

fine_tune_anchored() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.LunarLander.fine_tune_landers "$file" --seed 1 -a --gravity "-2.0" --epochs 15 | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

while IFS= read -r file; do
    echo "$file"
    # Fine-tune without anchors
    fine_tune_naive "$file" results/lander/naive/paths.txt &
    sleep 1

    # Fine-tune with anchors
    fine_tune_anchored "$file" results/lander/anchored/paths.txt &
    sleep 1
done < results/lander/original/paths.txt

wait
