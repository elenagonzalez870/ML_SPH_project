#!/usr/bin/env python
# coding: utf-8

# Elena Gonzalez Prieto
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
import argparse
import yaml
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import *

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, labels, reg_labels, data, transform=None, target_transform=None):
        self.labels = labels
        self.reg_labels = reg_labels
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        
        data = self.data[idx]
        label = self.labels[idx]
        reg_label = self.reg_labels[idx]

        data = torch.from_numpy(data).type(torch.float)
        label = torch.tensor(label)
        reg_label = torch.from_numpy(reg_label).type(torch.float)

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
                            
        return data, (label, reg_label)


class MultiTaskNeuralNetwork(nn.Module):
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
        )

        # Classification head
        self.class_head = nn.Linear(128, 4)
        self.log_sigma_class = nn.Parameter(torch.tensor(0.0))

        # Regression head
        self.reg_head = nn.Linear(128, 3)
        self.log_sigma_reg = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_relu_stack(x)
        class_out = self.class_head(x)
        reg_out = self.reg_head(x)
        reg_out_fractions = F.softmax(reg_out, dim=-1)
        return class_out, reg_out_fractions

class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2, device="cpu"):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True, device=device)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(self.params[i] ** 2)
        return loss_sum   

def train(dataloader, model, loss_fn, reg_loss_fn, optimizer, train_mean, train_std, cfg, awl, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.train()

    train_loss, correct = 0, 0
    y_true_class, y_true_reg = [], []
    y_pred_class, y_pred_reg = [], []
    initial_total_massess = []

    for X, (true_class, true_reg) in dataloader:
        X = X.to(device)
        true_class = true_class.to(device)
        true_reg = true_reg.to(device)

        pred_class, pred_reg = model(X)

        # Get initial (unnormalized masses)
        mass1i = np.exp((X[:, 3].detach().cpu().numpy() * train_std[3] + train_mean[3]))
        mass2i = np.exp((X[:, 4].detach().cpu().numpy() * train_std[4] + train_mean[4]))

        initial_total_masses = mass1i + mass2i
        initial_total_massess.append(initial_total_masses)

        # Calculating the loss for each task 
        loss_class = loss_fn(pred_class, true_class.argmax(dim=1))
        loss_reg = reg_loss_fn(pred_reg, true_reg)

        # Combining the loss 
        loss = awl(loss_class, loss_reg)
        train_loss += loss.item() * X.size(0) 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred_class.argmax(dim=1) == true_class.argmax(dim=1)).type(torch.float).sum().item()
        y_true_class.extend(true_class.argmax(dim=1).detach().cpu().numpy())
        y_pred_class.extend(pred_class.argmax(dim=1).detach().cpu().numpy())

        y_true_reg.append(true_reg.detach().cpu().numpy())  
        y_pred_reg.append(pred_reg.detach().cpu().numpy())

    train_loss /= size
    correct /= size
    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)

    # stack into (N, 3) arrays
    y_true_reg = np.vstack(y_true_reg)
    y_pred_reg = np.vstack(y_pred_reg)

    # assume y_true and y_pred are torch tensors
    errors = np.abs(y_pred_reg - y_true_reg) 
    median_absolute_error = np.median(errors, axis=0)

    # Transform true labels 
    initial_total_masses = np.concatenate(initial_total_massess)
    true_mass1 = y_true_reg[:, 0] * initial_total_masses 
    true_mass2 = y_true_reg[:, 1] * initial_total_masses
    true_mass_ejec = y_true_reg[:, 2] * initial_total_masses

    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred_reg[:, 0] * initial_total_masses
    predicted_mass2 = y_pred_reg[:, 1] * initial_total_masses
    predicted_mass_ejec = y_pred_reg[:, 2] * initial_total_masses

    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 
    median_abs_error_mejec = np.median(np.abs(predicted_mass_ejec - true_mass_ejec)) 
    
    #-- Error metric 3: Relative errors for cases where at least one star survives
    median_rel_error_m1 = np.median(np.abs(predicted_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ]) / true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(predicted_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ]) / true_mass2[true_mass2 != 0.])
    median_rel_error_m_ejec = np.median(np.abs(predicted_mass_ejec[true_mass_ejec != 0.] - true_mass_ejec[true_mass_ejec != 0. ]) / true_mass_ejec[true_mass_ejec != 0.])

    train_median_abs_errors = [median_abs_error_m1, median_abs_error_m2, median_abs_error_mejec]
    train_median_rel_errors = [median_rel_error_m1, median_rel_error_m2, median_rel_error_m_ejec]

    return train_loss, correct, balanced_acc, train_median_abs_errors, train_median_rel_errors

def test(dataloader, model, loss_fn, reg_loss_fn, train_mean, train_std, cfg, awl, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.eval()
    val_loss, correct = 0, 0
    class_val_loss, reg_val_loss = 0, 0
    y_true_class, y_true_reg = [], []
    y_pred_class, y_pred_reg = [], []
    initial_total_massess = []

    with torch.no_grad():
        for X, (true_class, true_reg) in dataloader:
            X = X.to(device)
            true_class = true_class.to(device)
            true_reg = true_reg.to(device)

            pred_class, pred_reg = model(X)

            # Get initial (unnormalized masses)
            mass1i = np.exp((X[:, 3].detach().cpu().numpy() * train_std[3] + train_mean[3]))
            mass2i = np.exp((X[:, 4].detach().cpu().numpy() * train_std[4] + train_mean[4]))

            initial_total_masses = mass1i + mass2i
            initial_total_massess.append(initial_total_masses)

            # Calculating the loss for each task 
            loss_class = loss_fn(pred_class, true_class.argmax(dim=1))
            loss_reg = reg_loss_fn(pred_reg, true_reg)

            # Combining the loss 
            loss = awl(loss_class, loss_reg)
            val_loss += loss.item() * X.size(0) 
            class_val_loss += loss_class.item() * X.size(0) 
            reg_val_loss += loss_reg.item() * X.size(0) 

            correct += (pred_class.argmax(dim=1) == true_class.argmax(dim=1)).type(torch.float).sum().item()

            y_true_class.extend(true_class.argmax(dim=1).detach().cpu().numpy())
            y_pred_class.extend(pred_class.argmax(dim=1).detach().cpu().numpy())

            y_true_reg.append(true_reg.detach().cpu().numpy())  
            y_pred_reg.append(pred_reg.detach().cpu().numpy())
            
    val_loss /= size
    class_val_loss /= size 
    reg_val_loss /= size
    correct /= size
    balanced_acc = balanced_accuracy_score(y_true_class, y_pred_class)

    #-- Error metric 1: Absolute Errors in the mass ratios 
    y_true_reg = np.vstack(y_true_reg) # stack into (N, 3) arrays
    y_pred_reg = np.vstack(y_pred_reg) 

    median_absolute_error = np.median(np.abs(y_pred_reg - y_true_reg), axis=0)

    # Transform true labels 
    initial_total_masses = np.concatenate(initial_total_massess)
    true_mass1 = y_true_reg[:, 0] * initial_total_masses 
    true_mass2 = y_true_reg[:, 1] * initial_total_masses
    true_mass_ejec = y_true_reg[:, 2] * initial_total_masses

    #-- Error metric 2: Absolute Errors in Mass 1 
    predicted_mass1 = y_pred_reg[:, 0] * initial_total_masses
    predicted_mass2 = y_pred_reg[:, 1] * initial_total_masses
    predicted_mass_ejec = y_pred_reg[:, 2] * initial_total_masses
    predicted_masses = np.stack((predicted_mass1, predicted_mass2), axis=1)


    median_abs_error_m1 = np.median(np.abs(predicted_mass1 - true_mass1)) 
    median_abs_error_m2 = np.median(np.abs(predicted_mass2 - true_mass2)) 
    median_abs_error_mejec = np.median(np.abs(predicted_mass_ejec - true_mass_ejec)) 
    
    #-- Error metric 3: Relative errors for cases where at least one star survives
    median_rel_error_m1 = np.median(np.abs(predicted_mass1[true_mass1 != 0.] - true_mass1[true_mass1 != 0. ]) / true_mass1[true_mass1 != 0.])
    median_rel_error_m2 = np.median(np.abs(predicted_mass2[true_mass2 != 0.] - true_mass2[true_mass2 != 0. ]) / true_mass2[true_mass2 != 0.])
    median_rel_error_m_ejec = np.median(np.abs(predicted_mass_ejec[true_mass_ejec != 0.] - true_mass_ejec[true_mass_ejec != 0. ]) / true_mass_ejec[true_mass_ejec != 0.])

    val_median_abs_errors = [median_abs_error_m1, median_abs_error_m2, median_abs_error_mejec]
    val_median_rel_errors = [median_rel_error_m1, median_rel_error_m2, median_rel_error_m_ejec]
    
    return val_loss, class_val_loss, reg_val_loss, balanced_acc, correct, val_median_abs_errors, val_median_rel_errors, y_pred_class, y_pred_reg


def main():

    # Simple argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None, help='Path to config YAML file')
    args = parser.parse_args()

    # Get today's date as string
    date_str = datetime.datetime.now().strftime("%m%d")
    
    # If config file provided, load it. Otherwise use wandb sweep
    if args.config:
        # CONFIG MODE: Load from YAML file
        print(f"Loading config from {args.config}")
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize wandb with config
        run = wandb.init(
            entity=config['logging']['entity'],
            project=config['logging']['project_name'],
            config=config
        )
        cfg = wandb.config
    else:
        # SWEEP MODE: wandb will provide config
        print("Using wandb sweep config")
        run = wandb.init(
            entity="elena-gonzalez-northwestern-university",
            project="ML_SPH_MultiTaskNN",
        )
        cfg = wandb.config

    # Generate run name AFTER init
    run_id = wandb.run.id
    wandbname = f"multitask_{date_str}_{run_id}"
    wandb.run.name = wandbname
    
    #--- Load and prepare data ---#
    data = np.load('../data_splits_splot22f_1008.npz')
    # y_data = [classification labels, M1,f/Mtot,i, M2,f/Mtot,i, Mejec,f/Mtot,i]
    # x_data = [Age, Rp, Vinf, Mass1, Mass2]

    X_train = data['X_train']
    y_train_class = data['y_train'][:, :1].flatten().astype(int)
    y_train_reg = data['y_train'][:, 1:]

    X_val = data['X_val']
    y_val_class = data['y_val'][:, :1].flatten().astype(int)
    y_val_reg = data['y_val'][:, 1:]

    X_test = data['X_test']
    y_test_class = data['y_test'][:, :1].flatten().astype(int)
    y_test_reg = data['y_test'][:, 1:]

    # Standard Normalize the data 
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Define the transform
    transform = None
    num_classes = 4
    target_transform = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, y, value=1))

    # Create datasets
    train_dataset = CustomDataset(labels=y_train_class, reg_labels=y_train_reg, data=X_train, 
                                 transform=transform, target_transform=target_transform)
    val_dataset = CustomDataset(labels=y_val_class, reg_labels=y_val_reg, data=X_val, 
                               transform=transform, target_transform=target_transform)
    test_dataset = CustomDataset(labels=y_test_class, reg_labels=y_test_reg, data=X_test, 
                                transform=transform, target_transform=target_transform)

    g = torch.Generator()
    g.manual_seed(42)

    # Use batch_size from config
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data["batch_size"], shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data["batch_size"], shuffle=False, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, generator=g)

    # Get class weights for loss function 
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_class), y=y_train_class)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize model
    model = MultiTaskNeuralNetwork()
    model = model.float()

    # Set loss function and optimizer based on config
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    reg_loss_fn = nn.MSELoss()

    if cfg.training["optimizer"] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training["learning_rate"], weight_decay=cfg.training["weight_decay"])
    elif cfg.training["optimizer"] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training["learning_rate"], momentum=0.9, weight_decay=cfg.training["weight_decay"])

    if cfg.training["scheduler"] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.training["scheduler_factor"], 
                                     patience=cfg.training["scheduler_patience"], min_lr=cfg.training["min_lr"])
    elif cfg.training["scheduler"] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training["epochs"]) 

    # Training loop
    best_val_score = np.inf
    best_model_state = None

    awl = AutomaticWeightedLoss(2).to(device)

    for t in range(cfg.training["epochs"]):   
        train_loss, correct, balanced_acc, train_median_abs_errors, train_median_rel_errors = train(train_dataloader, model, loss_fn, reg_loss_fn, optimizer, train_mean, train_std, cfg, awl, device)
        val_loss, class_val_loss, reg_val_loss, val_balanced_acc, val_correct, val_median_abs_errors, val_median_rel_errors, _ , _ = test(val_dataloader, model, loss_fn, reg_loss_fn, train_mean, train_std, cfg, awl, device)

        # Log metrics to wandb
        wandb.log({
            # losses
            "train_loss": train_loss,
            "val_loss": val_loss, 
            "class_val_loss": class_val_loss, 
            "reg_val_loss":reg_val_loss, 

            # classification accuracies 
            "train_acc": 100*correct, 
            "train_balanced_acc": 100*balanced_acc,
            "val_acc": 100*val_correct, 
            "val_balanced_acc": 100*val_balanced_acc,  

            # regression errors 
            "train_abs_error_m1": train_median_abs_errors[0], 
            "train_abs_error_m2": train_median_abs_errors[1], 
            "train_rel_error_m1": train_median_rel_errors[0], 
            "train_rel_error_m2": train_median_rel_errors[1], 
            "val_abs_error_m1": val_median_abs_errors[0], 
            "val_abs_error_m2": val_median_abs_errors[1], 
            "val_rel_error_m1": val_median_rel_errors[0], 
            "val_rel_error_m2": val_median_rel_errors[1], 

            # parameter
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": t
        })

        if val_loss < best_val_score:
            best_val_score = val_loss
            best_model_state = model.state_dict()

        if cfg.training["scheduler"] == 'ReduceLROnPlateau':
            scheduler.step(val_loss)
      
        elif cfg.training["scheduler"] == 'CosineAnnealingLR':
            scheduler.step() 

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save best model
    model_name = f"../models/multitask_model_{date_str}_{run_id}.pt"
    checkpoint = {
        "model_state_dict": best_model_state,
        "train_mean": train_mean,
        "train_std": train_std,
    }
    torch.save(checkpoint, model_name)

    # Test evaluation
    _, _, _, test_balanced_acc, test_correct, test_median_abs_errors, test_median_rel_errors, y_pred_test_class, y_pred_test_reg = test(test_dataloader, model, loss_fn, reg_loss_fn, train_mean, train_std, cfg, awl, device)
    _, _, _, _, _, _, _, y_pred_val_class, y_pred_val_reg = test(val_dataloader, model, loss_fn, reg_loss_fn, train_mean, train_std, cfg, awl, device)
    _, _, _, _, _, _, _, y_pred_train_class, y_pred_train_reg = test(train_dataloader, model, loss_fn, reg_loss_fn, train_mean, train_std, cfg, awl, device)

    wandb.log({"test_balanced_acc": 100 * test_balanced_acc, "test_acc": test_correct * 100})


    wandb.log({
            # classification accuracies 
            "test_acc": 100*test_correct, 
            "test_balanced_acc": 100*test_balanced_acc,

            # regression errors 
            "test_abs_error_m1": test_median_abs_errors[0], 
            "test_abs_error_m2": test_median_abs_errors[1], 
            "test_rel_error_m1": test_median_rel_errors[0], 
            "test_rel_error_m2": test_median_rel_errors[1], 
        })


    # np.savez('results/Multitask_results_' + str(run_id) + '.npz',
    #         y_pred_train_class = y_pred_train_class,
    #         y_pred_train_reg = y_pred_train_reg, 
    #         y_pred_test_class = y_pred_test_class, 
    #         y_pred_test_reg = y_pred_test_reg,
    #         y_pred_val_class = y_pred_val_class, 
    #         y_pred_val_reg = y_pred_val_reg,
    # )
    # Create confusion matrix

    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig = gen_confusion_matrix(y_test_class, y_pred_test_class, 'NN', ax=axes)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)
    
    wandb.finish()


if __name__ == "__main__":
    main()
