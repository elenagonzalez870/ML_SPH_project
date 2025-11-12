#!/usr/bin/env python
# coding: utf-8

import torch
from torch import nn
from torch.utils.data import random_split, DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.transforms import ToTensor, Lambda
import matplotlib.colors as mcolors
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


import matplotlib.pyplot as plt
import wandb

import os
import datetime

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, labels, data, transform=None, target_transform=None):
        self.labels = labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        label = self.labels[idx]

        data = torch.from_numpy(data).type(torch.float)
        label = torch.tensor(label)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
                            
        return data, label


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        fractions = F.softmax(logits, dim=-1)  # Convert to probabilities
        return fractions


def train(dataloader, model, loss_fn, optimizer, train_mean, train_std, cfg, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.train()

    train_loss, median_absolute_error = 0, 0
    y_true = [] # Store labels for balanced accuracy
    y_pred = []  # Store labels for balanced accuracy
    initial_total_massess = []

    for X, y in dataloader:
        X = X.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        # Compute prediction error
        pred = model(X)

        # Get initial (unnormalized masses)
        mass1i = np.exp((X[:, 3].detach().cpu().numpy() * train_std[3] + train_mean[3]))
        mass2i = np.exp((X[:, 4].detach().cpu().numpy() * train_std[4] + train_mean[4]))

        initial_total_masses = mass1i + mass2i
        initial_total_massess.append(initial_total_masses)
        
        # Calculate overall loss
        main_loss = loss_fn(pred[:,:2], y[:,:2])
        ejec_loss = loss_fn(pred[:,2], y[:,2])
        
        loss = main_loss + cfg.training['auxiliary_weight'] * ejec_loss
        train_loss += loss.item() * X.size(0)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Store predictions and true labels
        y_true.append(y.detach().cpu().numpy())  
        y_pred.append(pred.detach().cpu().numpy())

    train_loss /= size
    
    # stack into (N, 2) arrays
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)

    # assume y_true and y_pred are torch tensors
    errors = np.abs(y_pred - y_true) 
    median_absolute_error = np.median(errors, axis = 0)

    # Transform true labels 
    initial_total_masses = np.concatenate(initial_total_massess)
    true_mass1 = (y_true[:,0]) * initial_total_masses 
    true_mass2 = (y_true[:,1]) * initial_total_masses
    true_mass_ejec = (y_true[:,2]) * initial_total_masses

    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred[:,0] * initial_total_masses
    predicted_mass2 = y_pred[:,1] * initial_total_masses
    predicted_mass_ejec = y_pred[:,2] * initial_total_masses

    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 
    median_abs_error_mejec = np.median(np.abs(predicted_mass_ejec - true_mass_ejec)) 
    
    #-- Error metric 3: Relative errors for cases where at least one star survives
    median_rel_error_m1 = np.median(np.abs(predicted_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ])/ true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(predicted_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ])/ true_mass2[true_mass2 != 0.])
    median_rel_error_m_ejec = np.median(np.abs(predicted_mass_ejec[true_mass_ejec != 0.] - true_mass_ejec[true_mass_ejec != 0. ])/ true_mass_ejec[true_mass_ejec != 0.])

    train_median_abs_errors = [median_abs_error_m1, median_abs_error_m2, median_abs_error_mejec]
    train_median_rel_errors = [median_rel_error_m1, median_rel_error_m2, median_rel_error_m_ejec]

    return train_loss, train_median_abs_errors, train_median_rel_errors
    
    


def test(dataloader, model, loss_fn, train_mean, train_std, cfg, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.eval()
    val_loss, median_absolute_error = 0, 0
    y_true = []
    y_pred = []
    initial_total_massess = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            pred = model(X)

            # Get initial (unnormalized masses)
            mass1i = np.exp((X[:, 3].detach().cpu().numpy() * train_std[3] + train_mean[3]))
            mass2i = np.exp((X[:, 4].detach().cpu().numpy() * train_std[4] + train_mean[4]))

            initial_total_masses = mass1i + mass2i
            initial_total_massess.append(initial_total_masses)

            # Calculate overall loss
            main_loss = loss_fn(pred[:,:2], y[:,:2])
            ejec_loss = loss_fn(pred[:,2], y[:,2])
            loss = main_loss + cfg.training['auxiliary_weight']* ejec_loss
            val_loss += loss.item() * X.size(0) 
                
            # Store predictions and true labels
            y_true.append(y.detach().cpu().numpy())  # Convert tensors to NumPy
            y_pred.append(pred.detach().cpu().numpy())

    val_loss /= size

    #-- Error metric 1: Absolute Errors in the mass ratios 
    y_true = np.vstack(y_true) # stack into (N, 2) arrays
    y_pred = np.vstack(y_pred) 

    median_absolute_error = np.median(np.abs(y_pred - y_true) , axis = 0)

    # Transform true labels 
    initial_total_masses = np.concatenate(initial_total_massess)
    true_mass1 = (y_true[:,0]) * initial_total_masses 
    true_mass2 = (y_true[:,1]) * initial_total_masses
    true_mass_ejec = (y_true[:,2]) * initial_total_masses

    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred[:,0] * initial_total_masses
    predicted_mass2 = y_pred[:,1] * initial_total_masses
    predicted_mass_ejec = y_pred[:,2] * initial_total_masses
    predicted_masses = np.stack((predicted_mass1, predicted_mass2), axis=1)


    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 
    median_abs_error_mejec = np.median(np.abs(predicted_mass_ejec - true_mass_ejec)) 
    
    #-- Error metric 3: Relative errors for cases where at least one star survives
    median_rel_error_m1 = np.median(np.abs(predicted_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ])/ true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(predicted_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ])/ true_mass2[true_mass2 != 0.])
    median_rel_error_m_ejec = np.median(np.abs(predicted_mass_ejec[true_mass_ejec != 0.] - true_mass_ejec[true_mass_ejec != 0. ])/ true_mass_ejec[true_mass_ejec != 0.])

    val_median_abs_errors = [median_abs_error_m1, median_abs_error_m2, median_abs_error_mejec]
    val_median_rel_errors = [median_rel_error_m1, median_rel_error_m2, median_rel_error_m_ejec]

    return val_loss, val_median_abs_errors, val_median_rel_errors, predicted_masses


def main():
    # Get today's date as string
    date_str = datetime.datetime.now().strftime("%m%d")
    
    # Initialize wandb - this will pull config from sweep
    run = wandb.init(
        entity="elena-gonzalez-northwestern-university",
        project="ML_SPH_NN_Regression",
    )
    
    # NOW get the config from wandb (set by sweep)
    cfg = wandb.config
    
    # Generate run name AFTER init
    run_id = wandb.run.id
    wandbname = f"regression_{date_str}_{run_id}"
    wandb.run.name = wandbname
    
    #--- Load and prepare data ---#
    data = np.load('../../data_splits_splot22f_1008.npz')

    X_train = data['X_train']
    y_train = data['y_train'][:, 1:]
    X_val = data['X_val']
    y_val = data['y_val'][:, 1:]
    X_test = data['X_test']
    y_test = data['y_test'][:, 1:]

    # Standard Normalize the data 
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std  # apply train stats
    X_val = (X_val - train_mean) / train_std  # apply train stats

    # Create separate datasets for train and test
    train_dataset = CustomDataset(labels = y_train, data = X_train, transform=None, target_transform = None)
    val_dataset = CustomDataset(labels = y_val, data = X_val, transform=None, target_transform = None)
    test_dataset = CustomDataset(labels = y_test, data = X_test, transform=None, target_transform = None)

    #Take this out for final iteration, it is here to make sure data is shuffled the same each time
    g = torch.Generator()
    g.manual_seed(42)

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data['batch_size'], shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data['batch_size'], shuffle=False, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, generator=g)

    # Initialize model
    model = NeuralNetwork()
    model = model.float()

    # Set loss function and optimizer based on config
    loss_fn = nn.L1Loss() 
    
    if cfg.training['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training['learning_rate'], weight_decay=cfg.training['weight_decay'])
    elif cfg.training['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training['learning_rate'], momentum=0.9, weight_decay=cfg.training['weight_decay'])

    
    if cfg.training['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.training['scheduler_factor'], patience=cfg.training['scheduler_patience'], min_lr=cfg.training['min_lr'])

    elif cfg.training['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training['epochs']) 

    # Training loop
    best_val_score = np.inf
    best_model_state = None

    for t in range(cfg.training['epochs']):    
        train_loss, train_median_abs_errors, train_median_rel_errors =  train(train_dataloader, model, loss_fn, optimizer, train_mean, train_std, cfg, device)
        val_loss, val_median_abs_errors, val_median_rel_errors, _ = test(val_dataloader, model, loss_fn, train_mean, train_std, cfg, device)

        # Log metrics to wandb
        wandb.log({
            "train_loss": train_loss, 
            "train_abs_error_m1": train_median_abs_errors[0], 
            "train_abs_error_m2": train_median_abs_errors[1], 
            "train_abs_error_mejec": train_median_abs_errors[2], 
            "train_rel_error_m1": train_median_rel_errors[0], 
            "train_rel_error_m2": train_median_rel_errors[1], 
            "train_rel_error_mejec": train_median_rel_errors[2],
            "val_loss": val_loss, 
            "val_abs_error_m1": val_median_abs_errors[0], 
            "val_abs_error_m2": val_median_abs_errors[1], 
            "val_abs_error_mejec": val_median_abs_errors[2], 
            "val_rel_error_m1": val_median_rel_errors[0], 
            "val_rel_error_m2": val_median_rel_errors[1], 
            "val_rel_error_mejec": val_median_rel_errors[2], 
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": t
        })


        if val_loss < best_val_score:
            best_val_score = val_loss
            best_model_state = model.state_dict()

        if cfg.training['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
      
        elif cfg.training['scheduler'] == 'CosineAnnealingLR':
            scheduler.step() 

    # Save best model
    model_name = f"../models/model_{date_str}_{run_id}.pt"
    checkpoint = {
        "model_state_dict": best_model_state,
        "train_mean": train_mean,
        "train_std": train_std,
    }
    torch.save(checkpoint, model_name)

    # Test evaluation
    test_loss, test_median_abs_errors, test_median_rel_errors, y_pred_test = test(test_dataloader, model, loss_fn, train_mean, train_std, cfg, device)
    _, _, _, y_pred_train = test(train_dataloader, model, loss_fn, train_mean, train_std, cfg, device)
    _, _, _, y_val_train = test(val_dataloader, model, loss_fn, train_mean, train_std, cfg, device)

    wandb.log({
            "test_loss": test_loss, 
            "test_abs_error_m1": test_median_abs_errors[0], 
            "test_abs_error_m2": test_median_abs_errors[1], 
            "test_rel_error_m1": test_median_rel_errors[0], 
            "test_rel_error_m2": test_median_rel_errors[1] 
        })

    # np.savez('results/NN_results_' + str(run_id) + '.npz',
    #         y_pred_train=y_pred_train,
    #         y_pred_val=y_val_train,
    #         y_pred_test=y_pred_test,
    # )
    
    wandb.finish()


if __name__ == "__main__":
    main()
