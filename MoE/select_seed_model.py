#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Commented with the help of ClaudeAI 
# Last modified: Dec 10, 2025

import torch
import numpy as np
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
import os
import wandb
from MOE import TaskSpecificMoE
import sys
import glob
import shutil

def clean(name):
    name = name.replace("MoE_", "MoE_model_")
    return os.path.splitext(name)[0] + '.pt'   # remove .pt or any extension

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

    y_true_class = y_true[:,:1].flatten().astype(int)
    true_reg = y_true[:,1:]

    model.eval()
    model.to(device)
    
    with torch.no_grad():
        pred_class, pred_reg = model(X_norm)

    y_pred_class = pred_class.argmax(dim=1)

    correct = (pred_class.argmax(dim=1) == y_true_class).type(torch.float).sum().item()
    accuracy = correct / len(y_true_class)

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
    true_masses = [true_mass1, true_mass2]
    
    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred_reg[:, 0] * initial_total_massess
    predicted_mass2 = y_pred_reg[:, 1] * initial_total_massess
    predicted_masses = [predicted_mass1, predicted_mass2]

    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 

    #-- Error metric 3: Relative errors for cases where at least one star survives
    median_rel_error_m1 = np.median(np.abs(predicted_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ]) / true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(predicted_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ]) / true_mass2[true_mass2 != 0.])

    median_abs_errors = [median_abs_error_m1, median_abs_error_m2]
    median_rel_errors = [median_rel_error_m1, median_rel_error_m2]


    return balanced_acc, accuracy, y_pred_class, median_abs_errors,  median_rel_errors, true_masses, predicted_masses

if __name__ == "__main__":

    # # --------------------------------------------------------
    # # 0. Redirect all printed output to file AND stdout
    # # --------------------------------------------------------
    os.makedirs("./best_models", exist_ok=True)
    sys.stdout = open("./best_models/seed_model_selection.txt", "w", buffering=1)

    # # --------------------------------------------------------
    # # 1. Move all seed models to NN_best_seed_models
    # # --------------------------------------------------------

    api = wandb.Api()
    sweep = api.sweep("elena-gonzalez-northwestern-university/ML_SPH_MoE/q6gwcciz")
    run_names = [run.name for run in sweep.runs if run.state == "finished"]
    print("Number of finished runs :", len(run_names))

    copied_path = './best_models/MoE_best_seed_models/'
    for run in run_names:
        clean_run = clean(run)
        shutil.copy('./models/' + clean_run, copied_path)

    print("Copied models over to ./best_models/MoE_best_seed_models/")

    # # --------------------------------------------------------
    # # 2. Load the data 
    # # --------------------------------------------------------

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Load and prepare data --- #
    data = np.load('../data_splits_splot22f_1215.npz')
    X_test = data['X_test']
    y_test = data['y_test']

    # --------------------------------------------------------
    # 3. Load all models and compute statistics
    # --------------------------------------------------------
    model_dirs = glob.glob('./best_models/MoE_best_seed_models/*.pt')

    results = []

    for model_name in model_dirs:
        
        model = TaskSpecificMoE()
        checkpoint = torch.load(model_name, map_location=torch.device('cpu'), weights_only=False )
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = [checkpoint["train_mean"], checkpoint["train_std"]]
        train_mean = scaler[0]
        train_std = scaler[1]

        # Scale datasets
        X_test_norm = (X_test - train_mean) / train_std
        X_test_norm   = torch.tensor(X_test_norm,   dtype=torch.float32, device = 'cpu')
        y_true_class = y_test[:,0]

        balanced_acc, acc, pred_class_test,  median_abs_errors, median_rel_errors, true_masses_test , predicted_masses_test  = evaluate_model(model, X_test, X_test_norm, y_test, device='cpu')
        
        # Get score per model 
        results.append({
                    'model_name': model_name, 
                    'balanced_accuracy': balanced_acc,
                    'accuracy': acc,
                    'medae1': median_abs_errors[0], 
                    'medae2': median_abs_errors[1], 
                    'rae1': median_rel_errors[0], 
                    'rae2': median_rel_errors[1], 
                    'class_y_pred': pred_class_test, 
                    'class_y_true':y_true_class,
                    'reg_y_pred': predicted_masses_test,
                    'reg_y_true':true_masses_test})


    # Create DataFrame
    df = pd.DataFrame(results)
    df.to_pickle('./seed_model_results.pkl')

    # Calculate normalized scores
    medae_min1 = df['medae1'].min()
    medae_max1 = df['medae1'].max()

    # Calculate normalized scores
    medae_min2 = df['medae2'].min()
    medae_max2 = df['medae2'].max()

    # Save normalization parameters
    np.savez('./seed_normalization_params.npz',
        medae_min1=medae_min1,
        medae_max1=medae_max1,
        medae_min2=medae_min2,
        medae_max2=medae_max2)

    scores = []
    for res in results:
        s = score(res['balanced_accuracy'], res['medae1'], medae_min1, medae_max1, res['medae2'], medae_min2, medae_max2)
        scores.append(s)

    df = pd.read_pickle('./seed_model_results.pkl')

    print(f"The test mean balanced accuracy is {np.mean(df['balanced_accuracy']*100):.1f} with one sigma {np.std(df['balanced_accuracy']*100):.3f}") 
    print(f"The test  mean accuracy is {np.mean(df['accuracy']*100):.1f} with one sigma {np.std(df['accuracy']*100):.1f}") 
    print(f"The test  mean MedAE1 is {np.mean(df['medae1']):.8f} with one sigma {np.std(df['medae1']):.8f}") 
    print(f"The test  mean MedAE2 is {np.mean(df['medae2']):.8f} with one sigma {np.std(df['medae2']):.8f}") 
    print(f"The test  mean medRAE1 is {np.mean(df['rae1']):.8f} with one sigma {np.std(df['rae1']):.8f}") 
    print(f"The test  mean medRAE2 is {np.mean(df['rae2']):.8f} with one sigma {np.std(df['rae2']):.8f}") 

    
    # --------------------------------------------------------
    # 3. Select the model with the best performance
    # --------------------------------------------------------
    idx = np.argmin(scores)
    model_names = df['model_name']
    best_model = model_names[idx] 
    row = df.loc[idx]
    print("Best model is ", best_model)



    print(f"The best test balanced accuracy is {row['balanced_accuracy']*100:.1f} ")
    print(f"The best test accuracy is {row['accuracy']*100:.1f}") 
    print(f"The best test MAE1 is {row['medae1']:.8f}") 
    print(f"The best test MAE2 is {row['medae2']:.8f}") 
    print(f"The best test RAE1 is {row['rae1']:.8f}") 
    print(f"The best test RAE2 is {row['rae2']:.8f}") 

    # --------------------------------------------------------
    # 4. Create symbolic link + copy model
    # --------------------------------------------------------
    
    symlink_path = "./best_models/MoE_best_model.pt"
    copied_path  = "./best_models/" + best_model.split('/')[-1]

    # Remove old symlink if it exists
    if os.path.islink(symlink_path):
        os.unlink(symlink_path)
    elif os.path.exists(symlink_path):
        os.remove(symlink_path)

    # Create symlink
    try:
        os.symlink(os.path.abspath(best_model), symlink_path)
        print(f"Created symbolic link: {symlink_path} -> {best_model}")
    except FileExistsError:
        print("Symlink already exists.")

    # Copy model
    shutil.copy(best_model, copied_path)
    print(f"Copied best model to {copied_path}")

    # --------------------------------------------------------
    # 5. Load best model and compute predictions
    # --------------------------------------------------------
    
    predictions_path  = './results/'

    # Save classification predictions
    np.savez(
        './results/class_results.npz',
        y_pred_test=row['class_y_pred'],
        y_true_test=row['class_y_true']
    )

    # Save regression predictions
    np.savez(
        './results/reg_results.npz',
        y_pred_test=row['reg_y_pred'],
        y_true_test=row['reg_y_true']
    )

    print(f"Saved predictions to {predictions_path}")
    sys.stdout.close()

