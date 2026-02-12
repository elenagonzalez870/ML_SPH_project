#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Commented with the help of ChatGPT
# Last modified: Jan 21, 2026
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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load and prepare data --- #
data = np.load('../../data_splits_splot22f_1215.npz')
X_val   = data['X_val']
y_val   = data['y_val'][:, :1].flatten().astype(int)


# # --------------------------------------------------------
# # 1. Redirect all printed output to file AND stdout
# # --------------------------------------------------------
os.makedirs("./best_models", exist_ok=True)
sys.stdout = open("../best_models/NN_sweep_results.txt", "w", buffering=1)


# --------------------------------------------------------
# 2. Load all models and compute statistics
# --------------------------------------------------------

api = wandb.Api()
sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_NN_Classification/ti00n5t1")
run_names = [run.name for run in sweep.runs if run.state == "finished"]
print("Number of finished runs :", len(run_names))
run_names_clean = []
for run in run_names:
    run_names_clean.append(clean(run))


val_balanced_accuracies = []

X_val_raw = data['X_val'] 

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
    X_val = (X_val_raw - train_mean) / train_std
    X_val = torch.tensor(X_val, dtype=torch.float32)

    with torch.no_grad():
        y_pred_val = model(X_val).argmax(dim=1).numpy()
        bal_acc = balanced_accuracy_score(y_val, y_pred_val)
        

    val_balanced_accuracies.append(bal_acc)
    
val_balanced_accuracies = np.array(val_balanced_accuracies)


# --------------------------------------------------------
# 3. Select the model with the best performance
# --------------------------------------------------------
idx = np.argmax(val_balanced_accuracies)
best_model_path = run_names_clean[idx]
print(f"Best model is {best_model_path} with validation balanced accuracy of {val_balanced_accuracies[idx]*100}")

#checking the checkpoint 
best_model_path_clean = clean(best_model_path)
checkpoint = torch.load('../models/' + best_model_path_clean, map_location=torch.device('cpu'), weights_only = False)
print("Compare this to the saved checkpoint best_val_balanced_acc : ", (checkpoint["best_val_balanced_acc"]))
print("Best model saved at epoch", (checkpoint["best_epoch"]))


# --------------------------------------------------------
# 4. Create symbolic link + copy model
# --------------------------------------------------------
copied_path  = "../best_models/" + best_model_path.split('/')[-1]

# Copy model
shutil.copy('../models/' + best_model_path, copied_path)
print(f"Copied best model to {copied_path}")


sys.stdout.close()

