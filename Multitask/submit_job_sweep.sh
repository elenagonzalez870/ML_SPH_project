#!/bin/bash
#SBATCH --job-name=multitask_wandb_sweep
#SBATCH --mail-user=elena.prieto@northwestern.edu 
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2  # 3 agents (0, 1, 2)
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate your virtual environment
source activate ML  # Change to your venv path

# Optional: Set wandb cache directory to scratch space
export WANDB_DIR=$HOME/wandb_logs
mkdir -p $WANDB_DIR

# Print some info
echo "Starting wandb agent on GPU: $CUDA_VISIBLE_DEVICES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Run training
# For CONFIG mode (single run with YAML config):
python Multitask_NN.py --config config.yaml
# wandb sweep sweep_config.yaml

# Run the wandb agent
wandb agent elena-gonzalez-northwestern-university/ML_SPH_MultiTaskNN/<sweep_id>