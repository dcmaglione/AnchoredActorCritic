#!/bin/bash

trap 'kill $(jobs -p)' EXIT

# Create directories and delete existing files if they exist
mkdir -p results/reacher/naive
mkdir -p results/reacher/anchored
rm -f results/reacher/naive/paths.txt
rm -f results/reacher/anchored/paths.txt

fine_tune_naive() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.Reacher.train_reacher -s_s 5000 -s 1 -e 15 -d 0.1 -a_n 0.01 -p "$file" | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

fine_tune_anchored() {
    local file=$1
    local output_file=$2
    
    PYTHONUNBUFFERED=1 python -m envs.Reacher.train_reacher -a -s_s 5000 -s 1 -e 15 -d 0.1 -a_n 0.01 -p "$file" | 
    tee >(grep "saving at" | tail -n 1 | sed 's/.*saving at //' >> "$output_file")
}

while IFS= read -r file; do
    echo "$file"
    # Fine-tune without anchors
    fine_tune_naive "$file" results/reacher/naive/paths.txt &
    sleep 1

    # Fine-tune with anchors
    fine_tune_anchored "$file" results/reacher/anchored/paths.txt &
    sleep 1
done < results/reacher/original/paths.txt

wait
