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

# # --------------------------------------------------------
# # 0. Move all seed models to NN_best_seed_models
# # --------------------------------------------------------

api = wandb.Api()
sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_NN_Regression/9uylazl5")
run_names = [run.name for run in sweep.runs if run.state == "finished"]
print("Number of finished runs :", len(run_names))

run_names_clean = []
copied_path = '../best_models/NN_best_seed_models/'
for run in run_names:
    clean_run = clean(run)
    shutil.copy('../models/' + clean_run, copied_path)

print("Copied models over to ../best_models/NN_best_seed_models/")


# # --------------------------------------------------------
# # 1. Load the data 
# # --------------------------------------------------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#--- Load and prepare data ---#
data = np.load('../../data_splits_splot22f_1215.npz')
X_test = data['X_test']
y_test = data['y_test'][:, 1:]

# # --------------------------------------------------------
# # 1. Redirect all printed output to file AND stdout
# # --------------------------------------------------------
os.makedirs("./best_models", exist_ok=True)
sys.stdout = open("../best_models/NN_seed_results.txt", "w", buffering=1)


# --------------------------------------------------------
# 2. Load all models and compute statistics
# --------------------------------------------------------
model_dirs = glob.glob('../best_models/NN_best_seed_models/*.pt')

test_median_abs_error_m1 = []
test_median_abs_error_m2 = []

test_median_rel_error_m1 = []
test_median_rel_error_m2 = []

all_pred_mass1_test = []
all_pred_mass2_test = []

all_true_mass1_test = []
all_true_mass2_test = []

for model_name in model_dirs:
    model = NeuralNetwork()
    checkpoint = torch.load(model_name, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    scaler = [checkpoint["train_mean"], checkpoint["train_std"]]
    train_mean = scaler[0]
    train_std = scaler[1]

    # Scale datasets
    X_test_norm = (X_test - train_mean) / train_std

    X_test_norm   = torch.tensor(X_test_norm,   dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_norm)

    # Rescale to save final masses 
    mass1i_test  = np.exp(X_test[:, 3])
    mass2i_test  = np.exp(X_test[:, 4])

    initial_total_masses_test = mass1i_test + mass2i_test # in Msun 

    pred_mass1_test = y_pred_test[:,0] * initial_total_masses_test
    pred_mass2_test =  y_pred_test[:,1] * initial_total_masses_test

    true_mass1_test = y_test[:,0] * initial_total_masses_test
    true_mass2_test = y_test[:,1] * initial_total_masses_test

    test_median_abs_error_m1.append(np.median(np.abs(pred_mass1_test - true_mass1_test)))
    test_median_abs_error_m2.append(np.median(np.abs(pred_mass2_test - true_mass2_test)))

    test_median_rel_error_m1.append(np.median(np.abs(pred_mass1_test[true_mass1_test != 0.] - true_mass1_test[true_mass1_test != 0. ])/ true_mass1_test[true_mass1_test != 0.]))
    test_median_rel_error_m2.append(np.median(np.abs(pred_mass2_test[true_mass2_test != 0.] - true_mass2_test[true_mass2_test != 0. ])/ true_mass2_test[true_mass2_test != 0.]))
    
    all_pred_mass1_test.append(pred_mass1_test)
    all_pred_mass2_test.append(pred_mass2_test)
    all_true_mass1_test.append(true_mass1_test)
    all_true_mass2_test.append(true_mass2_test)

print(f"The mean test abs error in m1 is {np.mean(test_median_abs_error_m1):.8f} with one sigma {np.std(test_median_abs_error_m1):.8f}")
print(f"The mean test abs error in m2 is {np.mean(test_median_abs_error_m2):.8f} with one sigma {np.std(test_median_abs_error_m2):.8f}")

print(f"The mean test rel error in m1 is {np.mean(test_median_rel_error_m1):.8f} with one sigma {np.std(test_median_rel_error_m1):.8f}")
print(f"The mean test rel error in m2 is {np.mean(test_median_rel_error_m2):.8f} with one sigma {np.std(test_median_rel_error_m2):.8f}")


# --------------------------------------------------------
# 3. Select the model with the best performance
# --------------------------------------------------------
idx = np.argmin(test_median_abs_error_m1)

print(f"The best test absolute error in m1 is {test_median_abs_error_m1[idx]:.8f} ")
print(f"The best test absolute error in m2 is {test_median_abs_error_m2[idx]:.8f} ")

print(f"The best test rel error in m1 is {test_median_rel_error_m1[idx]:.8f} ")
print(f"The best test rel error in m2 is {test_median_rel_error_m2[idx]:.8f} ")

best_model_path = model_dirs[idx]
print(f"Best model is {best_model_path}")


# --------------------------------------------------------
# 4. Create symbolic link + copy model
# --------------------------------------------------------
symlink_path = "../best_models/NN_best_model.pt"
copied_path  = "../best_models/" + best_model_path.split('/')[-1]

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


# --------------------------------------------------------
#5. Load best model and compute predictions
# --------------------------------------------------------

# Save predictions
predictions_path = os.path.join('../results/NN_results.npz')
np.savez(
    predictions_path,
    y_pred_test=[all_pred_mass1_test[idx], all_pred_mass2_test[idx]], 
    y_true_test=[all_true_mass1_test[idx], all_true_mass2_test[idx]])

print(f"Saved predictions to {predictions_path}")


sys.stdout.close()

