#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Commented with the help of ChatGPT
# Last modified: Dec 8, 2025
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda
import matplotlib.colors as mcolors
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import wandb

import os
import datetime

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('../../')
from utils import *
import random
import string
from NN_regression import NeuralNetwork
import shutil
import glob

def clean(name):
    name = name.replace("regression", "model")
    return os.path.splitext(name)[0] + '.pt'   # remove .pt or any extension


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--- Load and prepare data ---#
data = np.load('../../data_splits_splot22f_1215.npz')
X_val = data['X_val']
y_val = data['y_val'][:, 1:]

# # --------------------------------------------------------
# # 1. Redirect all printed output to file AND stdout
# # --------------------------------------------------------
os.makedirs("./best_models", exist_ok=True)
sys.stdout = open("../best_models/NN_sweep_results.txt", "w", buffering=1)


# --------------------------------------------------------
# 2. Load all models and compute statistics
# --------------------------------------------------------
api = wandb.Api()
sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_NN_Regression/yyufwtru")
run_names = [run.name for run in sweep.runs if run.state == "finished"]
print("Number of finished runs :", len(run_names))
run_names_clean = []
for run in run_names:
    run_names_clean.append(clean(run))

median_abs_error_m1 = []
median_abs_error_m2 = []

for run in run_names_clean:
    model = NeuralNetwork()

    checkpoint = torch.load(
        '../models/' + run,
        map_location='cpu',
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    train_mean = checkpoint["train_mean"]
    train_std  = checkpoint["train_std"]

    # Normalize fresh copy each time
    X_val_norm = (X_val - train_mean) / train_std
    X_val_norm = torch.tensor(X_val_norm, dtype=torch.float32)

    with torch.no_grad():
        y_pred_val = model(X_val_norm)


    # Rescale to save final masses 
    mass1i_val  =  np.exp(X_val[:, 3])
    mass2i_val  =  np.exp(X_val[:, 4])

    initial_total_masses_val = mass1i_val + mass2i_val # in Msun 

    pred_mass1_val  =  y_pred_val[:,0] * initial_total_masses_val
    pred_mass2_val  =  y_pred_val[:,1] * initial_total_masses_val

    true_mass1_val= y_val[:,0] * initial_total_masses_val
    true_mass2_val= y_val[:,1] * initial_total_masses_val

    median_abs_error_m1.append(np.median(np.abs(pred_mass1_val - true_mass1_val)))
    median_abs_error_m2.append(np.median(np.abs(pred_mass2_val - true_mass2_val)))

# --------------------------------------------------------
# 3. Select the model with the best performance
# --------------------------------------------------------
idx = np.argmin(median_abs_error_m1)
best_model_name = run_names_clean[idx]
print(best_model_name)

print(f"The median absolute error in m1 is {median_abs_error_m1[idx]:.8f} ")
print(f"The median absolute error in m2 is {median_abs_error_m2[idx]:.8f} ")

#checking the checkpoint 
checkpoint = torch.load('../models/' + best_model_name, map_location=torch.device('cpu'), weights_only = False)
print("Compare with checkpoint statistics on wandb\n")
print("Best model saved at epoch", (checkpoint["best_epoch"]))

# --------------------------------------------------------
# 4. Create symbolic link + copy model
# --------------------------------------------------------
symlink_path = "../best_models/NN_best_sweep_model.pt"
copied_path  = "../best_models/" + best_model_name
best_model_path = glob.glob('../models/' + best_model_name)[0]

# Remove old symlink if it exists
if os.path.islink(symlink_path):
    os.unlink(symlink_path)
elif os.path.exists(symlink_path):
    os.remove(symlink_path)

# Create symlink
try:
    os.symlink(os.path.abspath(best_model_path), symlink_path)
    print(f"Created symbolic link: {symlink_path} -> {best_model_path}")
except FileExistsError:
    print("Symlink already exists.")

# Copy model
shutil.copy(best_model_path, copied_path)
print(f"Copied best model to {copied_path}")


sys.stdout.close()

