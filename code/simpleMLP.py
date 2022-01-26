import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import pickle 

class SimpleMLP(nn.Module): 

    """
    A simple MLP module with two hidden layers. 
    
    Args: 
        in_dimension: the dimension of the (flattened) input
        hidden_dims: the dimensions of the hidden layers
        out_dimension: the dimension of the output (10 for MNIST)

    """

    def __init__(self, in_dimension, hidden_dims, out_dimension): 
        super(SimpleMLP, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dimension,hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], out_dimension),
            nn.LogSoftmax(dim=1)
            )
        
    def forward(self, x): 
        x = x.view(x.shape[0], -1)
        out = self.net(x)
        return out 

def evaluate_model(model, data_loader): 
    """
    Evaluates the MLP model for a specific data loader (train test or val)

    Args: 
        model: MLP model to evaluate 
        data_loader: the dataloader of the set to evaluate 
    
    Returns: 
        accuracy: a float giving the average accuracy of dataset 
    """
    true_preds, count = 0., 0

    for x, y in data_loader: 
        preds = model(x)

        true_preds +=(preds.argmax(dim=-1) == y).sum()
        count += y.shape[0]

    return true_preds/ count 


def train_MLP(data_loader, in_dimension, hidden_dims, out_dimension, epochs, lr, seed): 
    """
    Trains the MLP model.      

    Args: 
        data_loader: a dictionary of 'train' 'test' and 'val' dataloaders

    """

    # set the seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # statistics of training
    val_accuracies = []
    train_accuracies = []
    train_losses = []
    
    best_val_acc = -1

    # define model, loss module and optimizer 
    model = SimpleMLP(in_dimension, hidden_dims, out_dimension)
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs): 

        # ----- TRAINING ------------

        model.train()

        # statistics of training
        true_preds, count, current_loss = 0., 0, 0.  

        for x,y in data_loader['train']: 
            optimizer.zero_grad()
            preds = model(x)

            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()

            # # statistics of training 
            current_loss += loss.item()
            # true_preds += (preds.argmax(dim=-1) == y).sum()
            count += y.shape[0]
        
        train_loss = current_loss / count 
        train_acc = evaluate_model(model, data_loader['train'])
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # ----- VALIDATION --------

        model.eval() 

        val_acc = evaluate_model(model, data_loader['val'])
        if val_acc > best_val_acc: 
            best_model = deepcopy(model)
            best_val_acc = val_acc 
        
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch+1:2d}] Training accuracy: {train_acc*100.0:05.2f}%, Validation accuracy: {val_acc*100.0:05.2f}%")


    test_acc = evaluate_model(best_model, data_loader['test'])
    print(f'Test accuracy {test_acc*100.0:05.2f}%')

    return best_model, val_accuracies, train_accuracies, test_acc, train_losses

