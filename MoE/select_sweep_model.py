#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Commented with the help of ClaudeAI 
# Last modified: Dec 10, 2025

import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import os
import wandb
from MOE import TaskSpecificMoE
import sys
import glob
import shutil

def score(balanced_acc, medae1, medae_min1, medae_max1, medae2, medae_min2, medae_max2, alpha=0.8, beta=0.10, eta=0.10):

    # Normalize regression error to [0, 1]
    if medae_max1 - medae_min1 > 0:
        normalized_medae1 = (medae1 - medae_min1) / (medae_max1 - medae_min1)
    else:
        normalized_medae1 = 0  # All models have same MAE

    # Normalize regression error to [0, 1]
    if medae_max2 - medae_min2 > 0:
        normalized_medae2 = (medae2 - medae_min2) / (medae_max2 - medae_min2)
    else:
        normalized_medae2 = 0  # All models have same MAE

    threshold = 0.005
    if medae2 < threshold:
        if medae1 < threshold:
            alpha += beta + eta 
            beta = 0 
            eta = 0 
        else:
            beta += eta
            eta = 0 
    else:
        if medae1 < threshold:
            eta += beta
            beta = 0 
    
    return alpha * (1 - balanced_acc) + beta * normalized_medae1 + eta * normalized_medae2


def evaluate_model(model, X, X_norm, y_true, device='cpu'):

    y_true_class = y_true[:, :1].flatten().astype(int)
    true_reg = y_true[:, 1:]

    model.eval()
    model.to(device)
    
    with torch.no_grad():
        pred_class, pred_reg = model(X_norm)

    y_pred_class = pred_class.argmax(dim=1)

    # Rescale to save final masses 
    mass1i  = (np.exp(X[:, 3]))
    mass2i  = (np.exp(X[:, 4]))

    initial_total_massess = mass1i + mass2i

    y_true_reg = true_reg
    y_pred_reg = pred_reg

    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)

    #-- Error metric 1: Absolute Errors in the mass ratios 

    # Transform true labels 
    true_mass1 = y_true_reg[:, 0] * initial_total_massess 
    true_mass2 = y_true_reg[:, 1] * initial_total_massess

    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred_reg[:, 0] * initial_total_massess
    predicted_mass2 = y_pred_reg[:, 1] * initial_total_massess

    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 

    return balanced_acc, median_abs_error_m1, median_abs_error_m2

def clean(name):
    name = name.replace("MoE_", "MoE_model_")
    return os.path.splitext(name)[0] + '.pt'   # remove .pt or any extension

if __name__ == "__main__":

    if not glob.glob('./model_results.csv'):
        # Get list of models from sweep 
        
        sys.stdout = open("./best_models/model_selection.txt", "w", buffering=1)

        api = wandb.Api()
        sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_MoE/wljxyian")
        run_names = [run.name for run in sweep.runs if run.state == "finished"]
        print("Number of finished runs :", len(run_names))
        run_names_clean = []
        for run in run_names:
            run_names_clean.append(clean(run))

        #--- Load and prepare data ---#
        data = np.load('../data_splits_splot22f_1215.npz')
        X_val = data['X_val']
        y_val = data['y_val']

        results = []
      
        for run in run_names_clean:
            print("RUN : ", run)
            model = TaskSpecificMoE()
            checkpoint = torch.load('./models/' + run, map_location=torch.device('cpu'), weights_only=False )
            model.load_state_dict(checkpoint["model_state_dict"])
            scaler = [checkpoint["train_mean"], checkpoint["train_std"]]
            train_mean = scaler[0]
            train_std = scaler[1]

            # Scale datasets
            X_val_norm = (X_val - train_mean) / train_std
            X_val_norm   = torch.tensor(X_val_norm,   dtype=torch.float32, device = 'cpu')

            balanced_acc, median_abs_error_m1, median_abs_error_m2 = evaluate_model(model, X_val, X_val_norm, y_val, device='cpu')
            # Get score per model 
            results.append({
                        'model_name': run, 
                        'balanced_accuracy': balanced_acc,
                        'medae1': median_abs_error_m1, 
                        'medae2': median_abs_error_m2})

            print(f"balanced_accuracy: {balanced_acc}, medae1: {median_abs_error_m1}, medae2: {median_abs_error_m2}")

        # Create DataFrame
        df = pd.DataFrame(results)
        df.to_csv('model_results.csv', index=False)

        # Calculate normalized scores
        medae_min1 = df['medae1'].min()
        medae_max1 = df['medae1'].max()

        # Calculate normalized scores
        medae_min2 = df['medae2'].min()
        medae_max2 = df['medae2'].max()

        # Save normalization parameters
        np.savez('normalization_params.npz',
            medae_min1=medae_min1,
            medae_max1=medae_max1,
            medae_min2=medae_min2,
            medae_max2=medae_max2)

        scores = []
        for res in results:

            s = score(res['balanced_accuracy'], res['medae1'], medae_min1, medae_max1, res['medae2'], medae_min2, medae_max2)
            scores.append(s)

        sort = np.argsort(scores)
        print("The top 10 models in order are")
        print([run_names_clean[i] for i in sort[:10]])
        print([scores[i] for i in sort[:10]])
        sys.stdout.close()

    else:
        print("File exists")
        # Load results
        results = pd.read_csv('model_results.csv')

        # Load normalization params
        norm_params = np.load('normalization_params.npz')
        medae_min1 = norm_params['medae_min1']
        medae_max1 = norm_params['medae_max1']
        medae_min2 = norm_params['medae_min2']
        medae_max2 = norm_params['medae_max2']

        scores = []
        for _, res in results.iterrows():

            s = score(res['balanced_accuracy'], res['medae1'], medae_min1, medae_max1, res['medae2'], medae_min2, medae_max2)
            scores.append(s)

        sort = np.argsort(scores)
        print("The top 10 models in order are")

        for i in sort[:10]:
            print(f"model_name: {results['model_name'][i]}, balanced_accuracy: {results['balanced_accuracy'][i]}, medae1: {results['medae1'][i]}, medae2: {results['medae2'][i]}")
            print(scores[i] )
            
