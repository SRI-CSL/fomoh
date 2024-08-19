#!/bin/bash

# Check if the correct number of arguments is passed
if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <learning_rate> <batch_size> <method> <device>"
    exit 1
fi

# Assign the command-line arguments to variables
lr=$1
batch_size=$2
method=$3
device=$4

# Define the array of random seeds
seeds=(0 1 2 3 4)

# Loop over each seed
for seed in "${seeds[@]}"; do
  echo "Running training with seed $seed, learning rate $lr, batch size $batch_size, and method $method"
  python ../train.py --lr $lr --epochs 2000 --batch_size $batch_size --model logreg --device cuda:$device --method $method --seed $seed --no_wandb --save --save-dir ./best_results/ --run_to_end
done

echo "All training runs completed."
