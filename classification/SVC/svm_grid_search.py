#!/usr/bin/env python
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
    'kernel':['rbf', 'poly'], 
    'C':[0.1, 1, 10, 100, 1000, 10000], 
    'gamma':[0.001, 0.01, 0.1, 1], 
    'degree':[2,3]}

    
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
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    train_bal_acc = balanced_accuracy_score(y_train, y_pred_train)
    
    val_acc = accuracy_score(y_val, y_pred_val)
    val_bal_acc = balanced_accuracy_score(y_val, y_pred_val)
    
    test_acc = accuracy_score(y_test, y_pred_test)
    test_bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    
    print(f"\nTest Accuracy: {test_acc*100:.3f}%")
    print(f"Test Balanced Accuracy: {test_bal_acc*100:.3f}%")
    
    return y_pred_train, y_pred_val, y_pred_test

def save_results(grid_search, scaler, predictions, output_dir, results_dir):
    """Save model, scaler, and results"""
    import os
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"\nSaving results to {output_dir} and {results_dir}...")
    
    # Save best model
    model_path = os.path.join(output_dir, 'svm_best_model.pkl')
    joblib.dump(grid_search.best_estimator_, model_path)
    print(f"Saved best model to {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(output_dir, 'svm_scaler.pkl')
    joblib.dump(scaler, scaler_path)
    print(f"Saved scaler to {scaler_path}")
    
    # Save training info
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    results_path = os.path.join(output_dir, 'svm_training_info.pkl')
    joblib.dump(results, results_path)
    print(f"Saved training info to {results_path}")
    
    # Save predictions
    y_pred_train, y_pred_val, y_pred_test = predictions
    predictions_path = os.path.join(results_dir, 'svm_results.npz')
    np.savez(
        predictions_path,
        y_pred_train=y_pred_train,
        y_pred_val=y_pred_val,
        y_pred_test=y_pred_test
    )
    print(f"Saved predictions to {predictions_path}")

def main():

    results_dir = '/home/egp8636/b1095/ML_SPH/results/'
    output_dir = '/home/egp8636/b1095/ML_SPH/best_models/'
    data_file = '/home/egp8636/b1095/ML_SPH/data_splits_splot22f_1008.npz'

    """Main execution function"""

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_prepare_data(
        data_file
    )
    
    # Normalize data
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = normalize_data(
        X_train, X_val, X_test
    )
    
    # Create a split indicator array
    # -1 means "use for training", 0 means "use for validation"
    split_index = np.concatenate([
        np.full(len(X_train_scaled), -1),  # All train samples get -1
        np.zeros(len(X_val_scaled))         # All val samples get 0
    ])

    # Combine train and validation sets
    X_train_val = np.vstack([X_train_scaled, X_val_scaled])
    y_train_val = np.hstack([y_train, y_val])

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
    
    # Save results
    save_results(
        grid_search, scaler, predictions,
        output_dir, results_dir
    )
    

if __name__ == "__main__":
    main()
