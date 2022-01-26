import argparse
import numpy as np
import os
from copy import deepcopy
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import pickle 


def evaluate_osr_model(model, val_loader_known, val_loader_oe, threshold): 
    """
    This function evaluates the current osr model for known and unknown data. 
    Args: 
        model: the model to evaluate on 
        val_loader_known: the validation loader for the known data 
        val_loader_unknown: the validation loader for the unknown data 

    Returns: 
        performance: some performance metric that evaluates how good the model is
    """

    true_preds, count = 0., 0
    for known_set, unknown_set in zip(val_loader_known, val_loader_oe): 

        # TODO: find a way of evaluating the current model for OSR 
        ... 

    return None 


def finetune_MLP_OE(data_loader_known, data_loader_oe, model, epochs, lr, threshold, lamda=0.5): 
    """ 
    This function finetunes the mlp model by training on 'unknown' data. 
    Args: 
        data_loader_in: dataloader dict of the in-distribution or the known data 
        data_loader_out: dataloader dict of the out-of-distribution or the unknown data
        mlp_model: a (trained) mlp model 
        epochs: amount of epochs to train 
        lr: learning rate 
        threshold: threshold for evaluating the validation set during training 
        lamda: parameter on how much you want to weigh the unknown loss, default is 0.5
        
    Returns: 
        best_model:  the best finetuned model, evaluated on validation set
        train_losses: 

    References:
        Dan Hendrycks, Mantas Mazeika, Thomas Dietterich (2019)
        Deep Anomaly Detection with Outlier Exposure
        https://arxiv.org/abs/1812.0460

    """


    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # stats
    train_losses = []
    best_val_performance = -1
    val_performances = []

    for epoch in range(epochs):

        # ---- TRAINING ----
        model.train()

        # stats of training 
        running_loss = 0.
        count = 0

        for known_set, unknown_set in zip(data_loader_known['train'], data_loader_oe['train']): 
            optimizer.zero_grad()

            # concatenate the known and unknown data and define known-data targets            
            data = torch.cat((known_set[0], unknown_set[0]), 0)
            target = known_set[1]

            preds = model(data)

            loss = criterion(preds[:len(known_set[0])], target)

            # add to the loss: the cross entropy between the softmax and uniform distribution
            loss += lamda * -(preds[len(known_set[0]):].mean(1) - torch.logsumexp(preds[len(known_set[0]):], dim=1)).mean() 

            loss.backward()
            optimizer.step()

            # training stats 
            count += data.shape[0]
            running_loss += loss.detach() 

        train_loss = running_loss / count
        print(f'Epoch [{epoch+1}] Train loss {train_loss}')

        train_losses.append(train_loss)

        # --- VALIDATION --- 
        # TODO: implement this

        # model.eval() 
        # val_performance = evaluate_osr_model(model, data_loader_known['val'], data_loader_oe['val'])
        # if val_performance > best_val_performance: 
        #     best_model = deepcopy(model)
        #     best_val_performance = val_performance

        # val_performances.append(val_performance)

        # print(f"[Epoch {epoch+1:2d}] Training loss: {train_loss}, Validation performance: {val_performance}")
    
    model.eval()  
    best_model = model 

    return best_model, val_performances, train_losses 




