#!/bin/bash
#SBATCH --job-name=multitask
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=2G
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu

# Create necessary directories
mkdir -p logs
mkdir -p ../models

# Activate your virtual environment
source activate ML

# Set wandb cache directory
export WANDB_DIR=$HOME/wandb_logs
mkdir -p $WANDB_DIR

# Print diagnostic info
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"
echo "========================================="

# Run training
# For CONFIG mode (single run with YAML config):
python Multitask_NN.py --config config.yaml
