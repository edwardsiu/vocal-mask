"""Training Vocal-Mask model

usage: train.py [options] <data-root>

options:
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --checkpoint=<path>         Restore model from checkpoint path if given.
    -h, --help                  Show this help message and exit
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
from dataset import basic_collate, SpectrogramDataset
from hparams import hparams as hp
from lrschedule import noam_learning_rate_decay, step_learning_rate_decay, cyclic_cosine_annealing
import discordhook

global_step = 0
global_epoch = 0
global_test_step = 0
train_losses = []
valid_losses = []
use_cuda = torch.cuda.is_available()

def save_checkpoint(device, model, optimizer, step, checkpoint_dir, epoch):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_step{:09d}.pth".format(step))
    optimizer_state = optimizer.state_dict()
    global global_test_step
    global train_losses
    global valid_losses
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer_state,
        "global_step": step,
        "global_epoch": epoch,
        "global_test_step": global_test_step,
        "train_losses": train_losses,
        "valid_losses":valid_losses
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model, optimizer, reset_optimizer):
    global global_step
    global global_epoch
    global global_test_step
    global train_losses
    global valid_losses

    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    if not reset_optimizer:
        optimizer_state = checkpoint["optimizer"]
        if optimizer_state is not None:
            print("Load optimizer state from {}".format(path))
            optimizer.load_state_dict(checkpoint["optimizer"])
    global_step = checkpoint["global_step"]
    global_epoch = checkpoint["global_epoch"]
    #global_step = 0
    #global_epoch = 0
    global_test_step = checkpoint.get("global_test_step", 0)
    train_losses = checkpoint["train_losses"]
    valid_losses = checkpoint["valid_losses"]

    return model


def test_save_checkpoint():
    checkpoint_path = "checkpoints/"
    device = torch.device("cuda" if use_cuda else "cpu")
    model = build_model()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    global global_step, global_epoch, global_test_step
    save_checkpoint(device, model, optimizer, global_step, checkpoint_path, global_epoch)

    model = load_checkpoint(checkpoint_path+"checkpoint_step000000000.pth", model, optimizer, False)


def evaluate_model(device, model, path, checkpoint_dir, global_step):
    """evaluate model by generating sample spectrograms

    """

    mix_path = os.path.join(path, "mix")
    vox_path = os.path.join(path, "vox")
    files = os.listdir(mix_path)
    random.shuffle(files)
    print("Evaluating model...")
    paths = []
    for f in tqdm(files[:hp.num_evals]):
        mix_wav = load_wav(os.path.join(mix_path,f))
        vox_wav = load_wav(os.path.join(vox_path,f))
        S = model.generate_specs(device, mix_wav)
        Smix = S['stft']
        Svox = stft(vox_wav)
        mix_spec = scaled_mel_weight(Smix, 1, True)
        vox_spec = scaled_mel_weight(Svox, 1, True)
        ideal_mask = make_vocal_mask(Smix, Svox)
        file_id = f.split(".")[0]
        fig_path = os.path.join(checkpoint_dir, 'eval', f'step_{global_step:06d}_vox_spec_{file_id}.png')
        paths.append(fig_path)
        plt.figure()
        plt.subplot(221)
        plt.title("Mixture")
        show_spec(mix_spec)

        plt.subplot(222)
        plt.title("Ground Truth Vocal")
        show_spec(vox_spec)

        plt.subplot(223)
        plt.title("Generated Mask")
        show_spec(S["mask"]["vocals"].astype(np.bool))

        plt.subplot(224)
        plt.title("Target Mask")
        show_spec(ideal_mask)

        plt.tight_layout()
        plt.savefig(fig_path)
        plt.close('all')
    discordhook.send_files(paths, msg=f"Global Step: {global_step}")


def validation_step(device, model, iter_testloader, criterion):
    """check loss on validation set
    """

    model.eval()
    with torch.no_grad():
        x, y = next(iter_testloader)
        x, y = x.to(device), y.to(device)
        y_pred = model(x)
        loss = criterion(y_pred, y)
    return loss.item()

def validation(device, model, testloader, criterion):
    """check loss on entire validation set
    """

    model.eval()
    print("")
    running_loss = 0
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(testloader)):
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            running_loss += loss.item()
            avg_loss = running_loss / (i+1)
    return avg_loss

def get_learning_rate(global_step, n_iters):
    if hp.lr_schedule_type == 'fixed':
        current_lr = hp.initial_learning_rate
    elif hp.lr_schedule_type == 'step':
        current_lr = step_learning_rate_decay(hp.initial_learning_rate, 
                    global_step, hp.step_gamma, hp.lr_step_interval)
    elif hp.lr_schedule_type == 'one-cycle':
        max_iters = n_iters*hp.nepochs
        cycle_width = int(max_iters*hp.fine_tune)
        step_size = cycle_width//2
        if global_step < cycle_width:
            cycle = np.floor(1 + global_step/(2*step_size))
            x = abs(global_step/step_size - 2*cycle + 1)
            current_lr = hp.min_lr + (hp.max_lr - hp.min_lr)*max(0, (1-x))
        else:
            x = (max_iters - global_step)/(max_iters - cycle_width)
            current_lr = 0.01*hp.min_lr + 0.99*hp.min_lr*x
    elif hp.lr_schedule_type == 'cca':
        current_lr = cyclic_cosine_annealing(hp.min_lr, hp.max_lr,
                    global_step, hp.nepochs*n_iters, hp.M)
    else:
        current_lr = noam_learning_rate_decay(hp.initial_learning_rate, 
                    global_step, hp.noam_warm_up_steps)
    return current_lr


def train_loop(device, model, trainloader, testloader,  optimizer, checkpoint_dir, eval_dir):
    """Main training loop.

    """
    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.BCELoss()

    global global_step, global_epoch, global_test_step, train_losses, valid_losses
    n_iters = int(np.ceil(len(trainloader.dataset)/hp.batch_size))
    while global_epoch < hp.nepochs:
        iter_testloader = iter(testloader)
        running_loss = 0
        print(f"[Epoch {global_epoch}]")
        for i, (x, y) in enumerate(tqdm(trainloader)):
            model.train()
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            loss = criterion(y_pred, y)

            # calculate learning rate and update learning rate
            current_lr = get_learning_rate(global_step, n_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            loss.backward()

            # clip gradient norm
            #nn.utils.clip_grad_norm_(model.parameters(), hp.grad_norm)
            optimizer.step()

            train_loss = loss.item()
            running_loss += train_loss
            avg_loss = running_loss / (i+1)
            train_losses.append((global_step, loss.detach().item()))
            try:
                valid_loss = validation_step(device, model, iter_testloader, criterion)
            except StopIteration:
                iter_testloader = iter(testloader)
                valid_loss = validation_step(device, model, iter_testloader, criterion)
            valid_losses.append((global_step, valid_loss))

            if global_step % hp.send_loss_every_step == 0:
                discordhook.send_message(f"Step:{global_step}, lr:{current_lr:.6e}, training loss:{train_loss:.6f}, valid loss:{valid_loss:.6f}")

            # Evaluation
            if global_step != 0 and global_step % hp.eval_every_step == 0:
                with torch.no_grad():
                    evaluate_model(device, model, eval_dir, checkpoint_dir, global_step)

            global_step += 1

        # save checkpoint
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)

        # Validation
        avg_valid_loss = validation(device, model, testloader, criterion)
        msg = (f"Step:{global_step}, lr:{current_lr:.6e}, avg training loss:{avg_loss:.6f}, avg valid loss:{avg_valid_loss:.6f}")
        discordhook.send_message(msg)

    
        global_epoch += 1


if __name__=="__main__":
    args = docopt(__doc__)
    #print("Command line args:\n", args)
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path = args["--checkpoint"]
    data_root = args["<data-root>"]
    eval_dir = os.path.join(data_root, "eval")

    # make dirs, load dataloader and set up device
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.join(checkpoint_dir,'eval'), exist_ok=True)
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

    if hp.lr_schedule_type == 'fixed':
        print("using fixed learning rate of :{}".format(hp.initial_learning_rate))
    elif hp.lr_schedule_type == 'step':
        print("using exponential learning rate decay")
    elif hp.lr_schedule_type == 'noam':
        print("using noam learning rate decay")
    elif hp.lr_schedule_type == 'one-cycle':
        print('using one-cycle learning rate')
    elif hp.lr_schedule_type == 'cca':
        print('using cyclic cosine annealing learning rate')

    # load checkpoint
    if checkpoint_path is None:
        print("no checkpoint specified as --checkpoint argument, creating new model...")
        
    else:
        model = load_checkpoint(checkpoint_path, model, optimizer, True)
        print("loading model from checkpoint:{}".format(checkpoint_path))
        # set global_test_step to True so we don't evaluate right when we load in the model
        global_test_step = True

    # create dataloaders
    with open(os.path.join(data_root, 'spec_info.pkl'), 'rb') as f:
        spec_info = pickle.load(f)
    test_path = os.path.join(data_root, "test")
    with open(os.path.join(test_path, "test_spec_info.pkl"), 'rb') as f:
        test_spec_info = pickle.load(f)
    test_specs = test_spec_info
    train_specs = spec_info
    trainset = SpectrogramDataset(data_root, train_specs)
    testset = SpectrogramDataset(test_path, test_specs)
    random.shuffle(testset.metadata)
    if hp.validation_size is not None:
        testset.metadata = testset.metadata[:hp.validation_size]
    print(f"Training examples: {len(trainset)}")
    print(f"Validation examples: {len(testset)}")
    trainloader = DataLoader(trainset, collate_fn=basic_collate, shuffle=True, num_workers=2, batch_size=hp.batch_size)
    testloader = DataLoader(testset, collate_fn=basic_collate, shuffle=True, num_workers=2, batch_size=hp.test_batch_size)


    # main train loop
    try:
        train_loop(device, model, trainloader, testloader, optimizer, checkpoint_dir, eval_dir)
    except KeyboardInterrupt:
        print("Interrupted!")
        pass
    finally:
        print("saving model....")
        save_checkpoint(device, model, optimizer, global_step, checkpoint_dir, global_epoch)
    

