#!/bin/bash
#SBATCH --job-name=svc_gridsearch
#SBATCH --mail-user=elena.prieto@northwestern.edu 
#SBATCH --output=svc_output_ext.out
#SBATCH --error=svc_error_ext.out
#SBATCH --time=24:00:00
#SBATCH --partition=grail-std
#SBATCH --mem-per-cpu=3G
#SBATCH --ntasks=32
#SBATCH --account=b1095


# Activate your virtual environment
source activate ML  # Change to your venv path
python svm_grid_search_extension.py

