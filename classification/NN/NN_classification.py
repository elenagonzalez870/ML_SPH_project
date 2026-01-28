#!/usr/bin/env python
# coding: utf-8

# Author: Elena González Prieto
# Last modified: Jan 25th , 2026

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
import seaborn as sns

import sys
sys.path.append('../../')
from utils import *
import random
import string
from copy import deepcopy

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU

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
        label = torch.tensor(label, dtype=torch.long)

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
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.train()

    train_loss, correct = 0, 0
    y_true = []
    y_pred = []

    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        train_loss += loss.item() * X.size(0) 

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

         # Store predictions and true labels
        correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
        y_true.extend(y.detach().cpu().numpy())
        y_pred.extend(pred.argmax(dim=1).detach().cpu().numpy())

    train_loss /= size
    correct /= size
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    return train_loss, correct, balanced_acc


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    model = model.to(device)
    model.eval()

    val_loss, correct = 0, 0
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)
            val_loss += loss.item() * X.size(0) 
            
            # Store predictions and true labels
            correct += (pred.argmax(dim=1) == y).type(torch.float).sum().item()
            y_true.extend(y.detach().cpu().numpy())
            y_pred.extend(pred.argmax(dim=1).detach().cpu().numpy())
            
    val_loss /= size
    correct /= size
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    return val_loss, correct, balanced_acc, y_pred


def main():
    # Get today's date as string
    date_str = datetime.datetime.now().strftime("%m%d")
    
    # Initialize wandb - this will pull config from sweep
    run = wandb.init(
        entity="elena-gonzalez-northwestern-university",
        project="ML_SPH_NN_Classification",
    )
    
    # NOW get the config from wandb (set by sweep)
    cfg = wandb.config
    
    # Generate run name AFTER init
    run_id = wandb.run.id
    wandbname = f"classification_{date_str}_{run_id}"
    wandb.run.name = wandbname
    
    #--- Load and prepare data ---#
    data = np.load('../../data_splits_splot22f_1215.npz')

    X_train = data['X_train']
    y_train = data['y_train'][:, :1].flatten().astype(int)
    X_val = data['X_val']
    y_val = data['y_val'][:, :1].flatten().astype(int)
    X_test = data['X_test']
    y_test = data['y_test'][:, :1].flatten().astype(int)

    # Standard Normalize the data 
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)

    X_train = (X_train - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std
    X_val = (X_val - train_mean) / train_std

    # Define the transform
    transform = None
    num_classes = 4

    # Create datasets
    train_dataset = CustomDataset(labels=y_train, data=X_train, transform=transform)
    val_dataset = CustomDataset(labels=y_val, data=X_val, transform=transform)
    test_dataset = CustomDataset(labels=y_test, data=X_test, transform=transform)

    # Use batch_size from config
    g = torch.Generator()
    g.manual_seed(cfg.data["random_state"])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data['batch_size'], shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data['batch_size'], shuffle=False, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, generator=g)

    # Get class weights for loss function 
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize model
    set_seed(cfg.data["random_state"]) #set the seed
    model = NeuralNetwork()
    model = model.float()

    # Set loss function and optimizer based on config
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    if cfg.training['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training['learning_rate'], weight_decay=cfg.training['weight_decay'])
    elif cfg.training['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.training['learning_rate'], momentum=0.9, weight_decay=cfg.training['weight_decay'])

    if cfg.training['scheduler'] == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=cfg.training['scheduler_factor'], patience=cfg.training['scheduler_patience'], min_lr=cfg.training['min_lr'])

    elif cfg.training['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.training['epochs'], eta_min=cfg.training['min_lr']) 

    # Training loop
    best_val_score = 0.0
    best_model_state = None
    best_epoch = 0

    for t in range(cfg.training['epochs']):    
        train_loss, correct, balanced_acc = train(train_dataloader, model, loss_fn, optimizer, device)
        val_loss, val_correct, val_balanced_acc, _ = test(val_dataloader, model, loss_fn, device)

        # Log metrics to wandb
        wandb.log({
            "train_acc": 100*correct, 
            "train_balanced_acc": 100*balanced_acc, 
            "train_loss": train_loss,
            "val_acc": 100*val_correct, 
            "val_balanced_acc": 100*val_balanced_acc, 
            "val_loss": val_loss, 
            "lr": optimizer.param_groups[0]['lr'],
            "epoch": t
        })

        if val_balanced_acc > best_val_score:
            best_val_score = val_balanced_acc
            best_model_state = deepcopy(model.state_dict())
            best_epoch = t
            best_val_correct = val_correct
            best_val_balanced_acc = val_balanced_acc


        if cfg.training['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_balanced_acc)
      
        elif cfg.training['scheduler'] == 'CosineAnnealingLR':
            scheduler.step() 

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Test evaluation
    test_loss, test_correct, test_balanced_acc, y_pred_test = test(test_dataloader, model, loss_fn, device)

    wandb.log({"test_balanced_acc": 100 * test_balanced_acc, "test_acc": test_correct * 100})

    # Save best model
    model_name = f"../models/classification_{date_str}_{run_id}.pt"
    checkpoint = {
        "model_state_dict": best_model_state,
        "train_mean": train_mean,
        "train_std": train_std,
        "test_accuracy": test_correct * 100, 
        "test_balanced_accuracy":100 * test_balanced_acc,
        "best_val_acc": 100*best_val_correct, 
        "best_val_balanced_acc": 100*best_val_balanced_acc, 
        "best_epoch": best_epoch}

    wandb.log({
        "best_val_acc": 100*best_val_correct, 
        "best_val_balanced_acc": 100*best_val_balanced_acc, 
        "best_epoch": best_epoch})

    torch.save(checkpoint, model_name)

    # Create confusion matrix
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    ax = gen_confusion_matrix(y_test, y_pred_test, 'NN', ax=axes)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)
    
    wandb.finish()


if __name__ == "__main__":
    main()
