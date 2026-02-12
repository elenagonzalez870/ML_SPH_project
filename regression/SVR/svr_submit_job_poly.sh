#!/bin/bash
#SBATCH --job-name=poly_svr_gridsearch
#SBATCH --mail-user=elena.prieto@northwestern.edu 
#SBATCH --output=svr_output_poly.out
#SBATCH --error=svr_error_poly.out
#SBATCH --time=48:00:00
#SBATCH --partition=ciera-std
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=51
#SBATCH --account=b1094

source activate ML  # Change to your venv path

# Set the number of threads for numpy/sklearn
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Activate your virtual environment

python svr_grid_search_poly.py

