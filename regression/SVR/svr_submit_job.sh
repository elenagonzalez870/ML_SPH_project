#!/bin/bash
#SBATCH --job-name=svr_gridsearch
#SBATCH --mail-user=elena.prieto@northwestern.edu 
#SBATCH --output=svr_output.out
#SBATCH --error=svr_error.out
#SBATCH --time=6:00:00
#SBATCH --partition=ciera-std
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=32
#SBATCH --account=b1094


# Activate your virtual environment
source activate ML  # Change to your venv path
python svr_grid_search.py

