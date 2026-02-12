#!/usr/bin/env python
"""
SVM Grid Search for Multi-class Classification
Optimized for parallel execution on SLURM cluster
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.colors as mcolors
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
import joblib
from joblib import Parallel, delayed

def load_and_prepare_data(data_file):
    """Load and prepare training data"""
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    
    X_train = data['X_train']
    y_train = data['y_train'][:, 1:]
    X_val = data['X_val']
    y_val = data['y_val'][:, 1:]
    X_test = data['X_test']
    y_test = data['y_test'][:, 1:]
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Testing set: {X_test.shape}, {y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, X_val, X_test):
    """Normalize data using StandardScaler"""
    print("Normalizing data...")
    scaler = StandardScaler().fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def fit_grid(grid, X, y):
    grid.fit(X, y)
    return grid

def run_grid_search(X_train_val, y_train_val, ps):
    """Run grid search with predefined split"""
    print("\nStarting grid search...")
    # print(f"Using n_jobs={n_jobs} for parallel processing")
    
    # FIRST SEARCH (0)
    # parameters_m1 = {
    # 'kernel':['rbf'], 
    # 'C':[0.1, 1, 10, 100], 
    # 'gamma':[0.01, 0.1, 1, 10], 
    # 'epsilon': [0.001,0.01, 0.1]}

    # parameters_m2 = {
    # 'kernel':['rbf'], 
    # 'C':[0.1, 1, 10, 100], 
    # 'gamma':[0.01, 0.1, 1, 10], 
    # 'epsilon': [0.001,0.01, 0.1]}

    # parameters_mejecta = {
    # 'kernel':['rbf'], 
    # 'C':[0.1, 1, 10, 100, 1000], 
    # 'gamma':[0.01, 0.1, 1, 10], 
    # 'epsilon': [0.001,0.01, 0.1]}

    # SECOND SEARCH (1)--> WE KEPT THESE MODELS 
    # parameters_m1 = {
    # 'kernel':['rbf'], 
    # 'C':[1], 
    # 'gamma':[10, 50], 
    # 'epsilon': [0.00001, 0.0001, 0.001]}

    # parameters_m2 = {
    # 'kernel':['rbf'], 
    # 'C':[1], 
    # 'gamma':[10, 50], 
    # 'epsilon': [0.00001, 0.0001, 0.001]}

    # parameters_mejecta = {
    # 'kernel':['rbf'], 
    # 'C':[1], 
    # 'gamma':[10, 50], 
    # 'epsilon': [0.00001, 0.0001, 0.001]}

    # THIRD SEARCH (2)
    parameters_m1 = {
    'kernel':['rbf'], 
    'C':[1], 
    'gamma':[10], 
    'epsilon': [0.0000001, 0.000001, 0.00001]}

    parameters_m2 = {
    'kernel':['rbf'], 
    'C':[1], 
    'gamma':[10], 
    'epsilon': [0.0000001, 0.000001, 0.00001]}

    parameters_mejecta = {
    'kernel':['rbf'], 
    'C':[1], 
    'gamma':[10], 
    'epsilon': [0.0000001, 0.000001, 0.00001]}

    # Run grid search
    grid_search_m1 = GridSearchCV(
        SVR(), 
        parameters_m1, 
        scoring='neg_mean_absolute_error',
        cv=ps,
        n_jobs=17,
        verbose=2  # Show progress
    )
    grid_search_m2 = GridSearchCV(
        SVR(), 
        parameters_m2, 
        scoring='neg_mean_absolute_error',
        cv=ps,
        n_jobs=17,
        verbose=2  # Show progress
    )
    grid_search_mejecta = GridSearchCV(
        SVR(), 
        parameters_mejecta, 
        scoring='neg_mean_absolute_error',
        cv=ps,
        n_jobs=17,
        verbose=2  # Show progress
    )
    
    print("Fitting model...")
    start_time = datetime.now()
    # grid_search_m1.fit(X_train_val, y_train_val[:, 0])
    # grid_search_m2.fit(X_train_val, y_train_val[:, 1])
    # grid_search_mejecta.fit(X_train_val, y_train_val[:, 2])

    grids = [
    (grid_search_m1, y_train_val[:, 0]),
    (grid_search_m2, y_train_val[:, 1]),
    (grid_search_mejecta, y_train_val[:, 2]),]

    # Run all three grid searches at the same time
    grid_search_m1, grid_search_m2, grid_search_mejecta = Parallel(n_jobs=3)(
        delayed(fit_grid)(grid, X_train_val, y)
        for grid, y in grids
    )

    end_time = datetime.now()
    
    print(f"\nGrid search completed in {end_time - start_time}")
    print(f"Best parameters M1: {grid_search_m1.best_params_}")
    print(f"Best cross-validation score M1: {grid_search_m1.best_score_:.3f}")
    print(f"Best parameters M2: {grid_search_m2.best_params_}")
    print(f"Best cross-validation score M2: {grid_search_m2.best_score_:.3f}")
    print(f"Best parameters Mejecta: {grid_search_mejecta.best_params_}")
    print(f"Best cross-validation score Mejecta: {grid_search_mejecta.best_score_:.3f}")
    
    return grid_search_m1, grid_search_m2, grid_search_mejecta

def evaluate_model(models, svr_scaler, X_test, y_test):
    """Evaluate model on all datasets"""
    print("\nEvaluating model...")
    
    model_m1, model_m2, model_mejecta = models

    # Calculate metrics
    y_pred_m1 = model_m1.predict(X_test)
    y_pred_m2 = model_m2.predict(X_test)
    y_pred_mejecta = model_mejecta.predict(X_test)

    # Calculate absolute and relative errors for each mass component 
    X_test_unnormalized = svr_scaler.inverse_transform(X_test)
    mass1i = np.exp(X_test_unnormalized[:, 3] )
    mass2i = np.exp(X_test_unnormalized[:, 4] )
    print(mass1i, mass2i)

    initial_total_masses = mass1i + mass2i # in Msun 
    pred_mass1 = y_pred_m1 * initial_total_masses
    pred_mass2 = y_pred_m2 * initial_total_masses
    pred_ejec  = y_pred_mejecta * initial_total_masses

    true_mass1 = y_test[:,0] * initial_total_masses
    true_mass2 = y_test[:,1] * initial_total_masses
    true_ejec  = y_test[:,2] * initial_total_masses


    #--Error metric 1: Median Absolute Errors for the respective masses 
    median_abs_error_m1 = np.median(np.abs(pred_mass1 - true_mass1))
    median_abs_error_m2 = np.median(np.abs(pred_mass2 - true_mass2))
    median_abs_error_ejec = np.median(np.abs(pred_ejec - true_ejec))

    #--Error metric 2: Relative errors for cases where at least one star survives

    median_rel_error_m1 = np.median(np.abs(pred_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ])/ true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(pred_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ])/ true_mass2[true_mass2 != 0.])
    median_rel_error_m_ejec = np.median(np.abs(pred_ejec[true_ejec != 0.] - true_ejec[true_ejec != 0. ])/ true_ejec[true_ejec != 0.])

    print(f"Absolute Errors M1 [Msun]: {median_abs_error_m1:.4f}")
    print(f"Absolute Errors M2 [Msun]: {median_abs_error_m2:.4f}")
    print(f"Relative Errors M1,f  : {median_rel_error_m1:.4f}")
    print(f"Relative Errors M2,f: {median_rel_error_m2:.4f}")
    
    return pred_mass1, pred_mass2, true_mass1, true_mass2

def save_results(grid_search, scaler, predictions_train, predictions_val, predictions_test, output_dir, results_dir):
    """Save model, scaler, and results"""
    import os
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nSaving results to {output_dir} and {results_dir}...")
    
    # Save predictions
    pred_mass1_train, pred_mass2_train, true_mass1_train, true_mass2_train = predictions_train
    pred_mass1_val, pred_mass2_val, true_mass1_val, true_mass2_val = predictions_val
    pred_mass1_test, pred_mass2_test, true_mass1_test, true_mass2_test = predictions_test
    
    np.savez(f'{results_dir}/svr_results_rbf_2.npz',  
         y_pred_train=[pred_mass1_train, pred_mass2_train],
         y_pred_val=[pred_mass1_val, pred_mass2_val],
         y_pred_test=[pred_mass1_test, pred_mass2_test], 
         y_true_train=[true_mass1_train, true_mass2_train],
         y_true_val=[true_mass1_val, true_mass2_val],
         y_true_test=[true_mass1_test, true_mass2_test])

    grid_search_m1, grid_search_m2, grid_search_mejecta = grid_search
    save_obj_m1 = {
        'model': grid_search_m1.best_estimator_,
        'scaler': scaler, 
        'best_params': grid_search_m1.best_params_,
        'best_score': grid_search_m1.best_score_,
        'cv_results': grid_search_m1.cv_results_}

    joblib.dump(save_obj_m1, f"{output_dir}/svr_best_model_rbf_2_m1.pkl")

    save_obj_m2 = {
        'model': grid_search_m2.best_estimator_,
        'scaler': scaler, 
        'best_params': grid_search_m2.best_params_,
        'best_score': grid_search_m2.best_score_,
        'cv_results': grid_search_m2.cv_results_}

    joblib.dump(save_obj_m2, f"{output_dir}/svr_best_model_rbf_2_m2.pkl")

    save_obj_mejecta = {
        'model': grid_search_mejecta.best_estimator_,
        'scaler': scaler, 
        'best_params': grid_search_mejecta.best_params_,
        'best_score': grid_search_mejecta.best_score_,
        'cv_results': grid_search_mejecta.cv_results_}

    joblib.dump(save_obj_mejecta, f"{output_dir}/svr_best_model_rbf_2_ejecta.pkl")
    

def main():

    results_dir = '/home/egp8636/b1095/ML_SPH/regression/results/'
    output_dir = '/home/egp8636/b1095/ML_SPH/regression/best_models/'
    data_file = '/home/egp8636/b1095/ML_SPH/data_splits_splot22f_1215.npz'

    """Main execution function"""

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(
        data_file)
    
    # Normalize data
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
        X_train, X_val, X_test)
    
    # Create a split indicator array
    # -1 means "use for training", 0 means "use for validation"
    split_index = np.concatenate([
        np.full(len(X_train_scaled), -1),  # All train samples get -1
        np.zeros(len(X_val_scaled))         # All val samples get 0
    ])

    # Combine train and validation sets
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.concatenate([y_train, y_val], axis=0)

    # Create the predefined split
    ps = PredefinedSplit(test_fold=split_index)

    # Run grid search
    svr_m1, svr_m2, svr_ejecta = run_grid_search(
        X_train_val, y_train_val, ps)
    
    # Evaluate model
    predictions_train = evaluate_model([svr_m1.best_estimator_, svr_m2.best_estimator_, svr_ejecta.best_estimator_], 
    scaler, X_train_scaled, y_train)

    predictions_val = evaluate_model([svr_m1.best_estimator_, svr_m2.best_estimator_, svr_ejecta.best_estimator_],
    scaler, X_val_scaled, y_val)

    predictions_test = evaluate_model([svr_m1.best_estimator_, svr_m2.best_estimator_, svr_ejecta.best_estimator_],
    scaler, X_test_scaled, y_test)
    
    # Save results
    save_results(
        [svr_m1, svr_m2, svr_ejecta], scaler, predictions_train, predictions_val, predictions_test,
        output_dir, results_dir)
    

if __name__ == "__main__":
    main()
