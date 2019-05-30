"""Usage:
    lr_find.py [-h] <dataroot> <minlr> <maxlr>
"""

from docopt import docopt

import os
from os.path import dirname, join, expanduser
import random
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import pickle

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from adamW import AdamW

from audio import *
from model import build_model
from loss_function import nll_loss
from dataset import basic_collate, SpectrogramDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay

use_cuda = torch.cuda.is_available()

def train_loop(device, model, trainloader, optimizer, min_lr, max_lr):
    """Main training loop.

    """
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()

    global_step = 0
    train_losses = []
    model.train()
    n_iters = 100
    lrs = np.logspace(min_lr, max_lr, n_iters)
    for i, (x, y) in enumerate(tqdm(trainloader)):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[i]

        loss.backward()

        # clip gradient norm
        #nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
        optimizer.step()

        global_step += 1
        train_losses.append((lrs[i], loss.item()))
        if global_step == n_iters:
            return train_losses


if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    data_root = args["<dataroot>"]
    min_lr = int(args["<minlr>"][1:-1])
    max_lr = int(args["<maxlr>"][1:-1])
    

    # make dirs, load dataloader and set up device
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))


    # build model, create optimizer
    model = build_model().to(device)
    if hp.optimizer == 'adam':
        optimizer = AdamW(model.parameters(),
                          lr=hp.initial_learning_rate, betas=(
                              hp.adam_beta1, hp.adam_beta2),
                          eps=hp.adam_eps, weight_decay=hp.weight_decay,
                          amsgrad=hp.amsgrad)
    else:
        optimizer = optim.SGD(model.parameters(), 
                              lr=hp.initial_learning_rate, 
                              momentum=hp.momentum, 
                              weight_decay=hp.weight_decay, 
                              nesterov=hp.nesterov)


    # create dataloaders
    with open(os.path.join(data_root, 'spec_info.pkl'), 'rb') as f:
        spec_info = pickle.load(f)
    train_specs = spec_info
    trainset = SpectrogramDataset(data_root, train_specs)
    trainloader = DataLoader(trainset, collate_fn=basic_collate, shuffle=True, num_workers=6, batch_size=hp.batch_size)


    # main train loop
    try:
        losses = train_loop(device, model, trainloader, optimizer, min_lr, max_lr)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        plt.figure()
        x, y = zip(*losses)
        plt.plot(x, y)
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.xscale('log')
        plt.title('Learning Rate Find')
        plt.show()
    

