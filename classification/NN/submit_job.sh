#!/bin/bash
#SBATCH --job-name=wandb_sweep
#SBATCH --mail-user=elena.prieto@northwestern.edu 
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --error=logs/sweep_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=ciera-gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2  # 3 agents (0, 1, 2)
#SBATCH --account=b1094
#SBATCH --partition=ciera-gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust these for your HPC)
module load anaconda

# Activate your virtual environment
source activate ML  # Change to your venv path

# Set wandb API key (if not already set)
# export WANDB_API_KEY="your_api_key_here"

# Optional: Set wandb cache directory to scratch space
export WANDB_DIR=$HOME/wandb_logs
mkdir -p $WANDB_DIR

# Print some info
echo "Starting wandb agent on GPU: $CUDA_VISIBLE_DEVICES"
echo "Array task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"

# Run the wandb agent
wandb agent elena-gonzalez-northwestern-university/ML_SPH_NN_Classification/oxnxnedm