"""Extract vocals from waveform.

usage: generate.py [options] <checkpoint-path> <input-wav>

options:
    --output-dir=<dir>      Directory where to save output wav [default: generated].
    --sr=<sr>               Sample rate of generated waveform
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import os
from os.path import dirname, join, expanduser
import random
from tqdm import tqdm

import numpy as np
import librosa
import librosa.display
import librosa.output
import matplotlib.pyplot as plt

import torch

from audio import *
from model import build_model
from hparams import hparams as hp
from utils import resample

use_cuda = torch.cuda.is_available()

def _load(checkpoint_path):
    if use_cuda:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path,
                                map_location=lambda storage, loc: storage)
    return checkpoint


def load_checkpoint(path, model):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    model.load_state_dict(checkpoint["state_dict"])
    return model

def scale_wav(wav):
    return wav/(np.max(np.abs(wav)))

def generate_mono(device, model, path, output_dir, target_sr):
    wav = load_wav(path, offset=30, duration=5)
    wav = wav/np.max(np.abs(wav))
    S = model.generate_specs(device, wav)
    Smix = stft(wav)
    H, P, R = hpss_decompose(Smix)
    Hmel = scaled_mel_weight(H, hp.power["mix"], True)
    Pmel = scaled_mel_weight(P, hp.power["mix"], True)
    Rmel = scaled_mel_weight(R, hp.power["mix"], True)
    plt.figure()
    plt.subplot(221)
    plt.title('Harmonic')
    show_spec(Hmel)
    plt.subplot(222)
    plt.title('Percussive')
    show_spec(Pmel)
    plt.subplot(223)
    plt.title('Residual')
    show_spec(Rmel)
    plt.subplot(224)
    plt.title('Mask')
    show_spec(S["mask"]["vocals"])
    plt.tight_layout()
    plt.show()
 

if __name__=="__main__":
    args = docopt(__doc__)
    output_dir = args["--output-dir"]
    checkpoint_path = args["<checkpoint-path>"]
    input_path = args["<input-wav>"]
    target_sr = args["--sr"]

    if output_dir is None:
        output_dir = 'generated'
    if target_sr is None:
        target_sr = hp.sample_rate
    else:
        target_sr = int(target_sr)

    # make dirs, load dataloader and set up device
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device:{}".format(device))

    # build model, create optimizer
    model = build_model().to(device)
    
    # load checkpoint
    model = load_checkpoint(checkpoint_path, model)
    print("loading model from checkpoint:{}".format(checkpoint_path))

    generate_mono(device, model, input_path, output_dir, target_sr)
