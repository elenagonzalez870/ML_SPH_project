#!/usr/bin/env python
# Author: Elena González Prieto
# Last modified: Nov 17, 2025

"""
SVM Grid Search for Multi-class Classification
Optimized for parallel execution on SLURM cluster
"""

import numpy as np
import sys
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import joblib
from datetime import datetime


def load_and_prepare_data(data_file):
    """Load and prepare training data"""
    print(f"Loading data from {data_file}...")
    data = np.load(data_file)
    
    X_train = data['X_train']
    y_train = data['y_train'][:, :1]
    X_val = data['X_val']
    y_val = data['y_val'][:, :1]
    X_test = data['X_test']
    y_test = data['y_test'][:, :1]
    
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

def run_grid_search(X_train_val, y_train_val, ps, n_jobs=-1):
    """Run grid search with predefined split"""
    print("\nStarting grid search...")
    print(f"Using n_jobs={n_jobs} for parallel processing")
    
    # Define parameter grid
    parameters = {
    'kernel':['rbf'], 
    'C':[10000, 20000, 30000], 
    'gamma':[0.001, 0.01, 0.1, 1]}
    
    # Create base SVC
    svc = svm.SVC(decision_function_shape='ovr', class_weight='balanced')
    
    # Run grid search
    grid_search = GridSearchCV(
        svc, 
        parameters, 
        scoring='balanced_accuracy',
        cv=ps,
        n_jobs=n_jobs,
        verbose=2  # Show progress
    )
    
    print("Fitting model...")
    start_time = datetime.now()
    grid_search.fit(X_train_val, y_train_val)
    end_time = datetime.now()
    
    print(f"\nGrid search completed in {end_time - start_time}")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.3f}")
    
    return grid_search

def evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """Evaluate model on all datasets"""
    print("\nEvaluating model...")
    
    # Generate predictions
    y_pred_train = model.predict(X_train)
    y_pred_val   = model.predict(X_val)
    y_pred_test  = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    
    val_acc = accuracy_score(y_val, y_pred_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_pred_val)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    
    print(f"\nTest Accuracy: {test_acc*100:.3f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.3f}%")
    
    return y_pred_train, y_pred_val, y_pred_test, test_acc, test_bal_acc


def main():
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
    y_train_val = np.vstack([y_train, y_val]).ravel()

    # Create the predefined split
    ps = PredefinedSplit(test_fold=split_index)

    # Run grid search
    grid_search = run_grid_search(
        X_train_val, y_train_val, ps, n_jobs=-1
    )
    
    # Evaluate model
    predictions = evaluate_model(
        grid_search.best_estimator_,
        X_train_scaled, y_train,
        X_val_scaled, y_val,
        X_test_scaled, y_test
    )

    

if __name__ == "__main__":
    main()
