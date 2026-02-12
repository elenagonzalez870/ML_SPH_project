#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Commented with the help of ChatGPT
# Last modified: January 25th, 2026

import sys, os
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

from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import matplotlib.pyplot as plt
import wandb

import datetime
import glob
import shutil
import sys

sys.path.append('../../')
from utils import *
from NN_classification import NeuralNetwork


def clean(name):
    name = name.replace("model", "classification")
    return os.path.splitext(name)[0] + '.pt'   # remove .pt or any extension

# # --------------------------------------------------------
# # 0. Move all seed models to NN_best_seed_models
# # --------------------------------------------------------

api = wandb.Api()
sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_NN_Classification/2r3v9fmt")
run_names = [run.name for run in sweep.runs if run.state == "finished"]
print("Number of finished runs :", len(run_names))

copied_path = '../best_models/NN_best_seed_models/'

print(f"Looking at {len(run_names)} models")
for run in run_names:
    clean_run = clean(run)
    shutil.copy('../models/' + clean_run, copied_path)

print("Copied models over to ../best_models/NN_best_seed_models/")


# # --------------------------------------------------------
# # 1. Load the data 
# # --------------------------------------------------------

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load and prepare data --- #
data = np.load('../../data_splits_splot22f_1215.npz')
X_test  = data['X_test']
y_test  = data['y_test'][:, :1].flatten().astype(int)


# # --------------------------------------------------------
# # 1. Redirect all printed output to file AND stdout
# # --------------------------------------------------------
os.makedirs("./best_models", exist_ok=True)
sys.stdout = open("../best_models/NN_seed_results.txt", "w", buffering=1)


# --------------------------------------------------------
# 2. Load all models and compute statistics
# --------------------------------------------------------
model_dirs = glob.glob('../best_models/NN_best_seed_models/*.pt')

test_balanced_accuracies = []
test_accuracies = []
test_predictions = []

for model_name in model_dirs:
    model = NeuralNetwork()

    checkpoint = torch.load(model_name, map_location='cpu', weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    train_mean = checkpoint["train_mean"]
    train_std  = checkpoint["train_std"]

    # Scale datasets
    X_test_norm  = (X_test  - train_mean) / train_std

    X_test_norm  = torch.tensor(X_test_norm,  dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        # Compute predictions 
        y_pred_test  = model(X_test_norm).argmax(dim=1).cpu().numpy()

        # Compute balanced accuracies 
        test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)

        # Compute accuracies 
        test_acc = accuracy_score(y_test, y_pred_test)

    test_balanced_accuracies.append(test_bal_acc * 100)
    test_accuracies.append(test_acc * 100)
    test_predictions.append(y_pred_test)
    
test_balanced_accuracies = np.array(test_balanced_accuracies)
test_accuracies = np.array(test_accuracies)

print(f"The mean test accuracy is {np.mean(test_accuracies):.1f} with one sigma {np.std(test_accuracies):.1f}") 
print(f"The mean test balanced accuracy is {np.mean(test_balanced_accuracies):.1f} with one sigma {np.std(test_balanced_accuracies):.1f}")

# --------------------------------------------------------
# 3. Select the model with the best performance
# --------------------------------------------------------
idx = np.argmax(test_balanced_accuracies)
best_model_path = model_dirs[idx]
print(f"Best model is {best_model_path}")
print(f"Test accuracy is {test_accuracies[idx]:.1f}") 
print(f"Test balanced accuracy is {test_balanced_accuracies[idx]:.1f} ")


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
# 5. Save predictions
# --------------------------------------------------------

# Save predictions
predictions_path = '../results/NN_results.npz'
np.savez(predictions_path,
         y_pred_test=test_predictions[idx])

print(f"Saved predictions to {predictions_path}")


sys.stdout.close()

