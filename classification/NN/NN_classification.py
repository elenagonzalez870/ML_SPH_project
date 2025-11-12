#!/usr/bin/env python
# coding: utf-8

# Elena Gonzalez Prieto

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

from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

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

        pred = model(X)
        loss = loss_fn(pred, y.argmax(dim=1))
        train_loss += loss.item() * X.size(0) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct += (pred.argmax(dim=1) == y.argmax(dim=1)).type(torch.float).sum().item()
        y_true.extend(y.argmax(dim=1).detach().cpu().numpy())
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
            loss = loss_fn(pred, y.argmax(dim=1))
            val_loss += loss.item() * X.size(0) 
            correct += (pred.argmax(dim=1) == y.argmax(dim=1)).type(torch.float).sum().item()

            y_true.extend(y.argmax(dim=1).detach().cpu().numpy())
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
    data = np.load('../../data_splits_splot22f_1008.npz')

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
    target_transform = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(0, y, value=1))

    # Create datasets
    train_dataset = CustomDataset(labels=y_train, data=X_train, transform=transform, target_transform=target_transform)
    val_dataset = CustomDataset(labels=y_val, data=X_val, transform=transform, target_transform=target_transform)
    test_dataset = CustomDataset(labels=y_test, data=X_test, transform=transform, target_transform=target_transform)

    g = torch.Generator()
    g.manual_seed(42)

    # Use batch_size from config
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, generator=g)

    # Get class weights for loss function 
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize model
    model = NeuralNetwork()
    model = model.float()

    # Set loss function and optimizer based on config
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    if cfg.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.learning_rate, momentum=0.9, weight_decay=cfg.weight_decay)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=cfg.scheduler_factor, 
                                 patience=cfg.scheduler_patience, min_lr=cfg.min_lr)

    # Training loop
    best_val_score = 0.0
    best_model_state = None

    for t in range(cfg.epochs):    
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
            best_model_state = model.state_dict()

        scheduler.step(val_loss)

    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save best model
    model_name = f"model_{date_str}_{run_id}.pt"
    checkpoint = {
        "model_state_dict": best_model_state,
        "train_mean": train_mean,
        "train_std": train_std,
    }
    torch.save(checkpoint, model_name)

    # Test evaluation
    test_loss, test_correct, test_balanced_acc, y_pred_test = test(test_dataloader, model, loss_fn, device)
    _, _, _, y_pred_train = test(train_dataloader, model, loss_fn, device)
    _, _, _, y_pred_val = test(val_dataloader, model, loss_fn, device)

    wandb.log({"test_balanced_acc": 100 * test_balanced_acc, "test_acc": test_correct * 100})
    # np.savez('results/NN_results_' + str(run_id) + '.npz',
    #         y_pred_train=y_pred_train,
    #         y_pred_val=y_pred_val,
    #         y_pred_test=y_pred_test,
    # )
    # Create confusion matrix
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    fig = gen_confusion_matrix(y_test, y_pred_test, 'NN', ax=axes)
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close(fig)
    
    wandb.finish()


if __name__ == "__main__":
    main()
