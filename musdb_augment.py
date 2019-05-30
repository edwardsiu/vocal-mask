"""
Augment the musdb dataset

usage: python musdb_augment.py [options] <musdb-root>

options:
    -h, --help              Show help message.
"""

import musdb
import librosa
import librosa.output
import librosa.effects
import numpy as np
import os
from multiprocessing import Manager, Process
#from docopt import docopt
import sys
from tqdm import tqdm
from hparams import hparams as hp

sample_rate = 44100

def to_mono(wav):
    return librosa.to_mono(wav.T)

def pitch_shift(wav, steps):
    wav = to_mono(wav)
    return librosa.effects.pitch_shift(wav, sample_rate, n_steps=steps)

def rotate(wav, seconds):
    wav = to_mono(wav)
    offset = seconds*sample_rate
    return np.hstack([wav[offset:], wav[:offset]])

def save_tracks(mixture, vocals, path):
    mixture = np.clip(mixture/np.max(np.abs(mixture)), -1.0, 1.0)
    vocals = np.clip(vocals/np.max(np.abs(vocals)), -1.0, 1.0)
    os.makedirs(path, exist_ok=True)
    librosa.output.write_wav(os.path.join(path, "mixture.wav"), mixture, sample_rate)
    librosa.output.write_wav(os.path.join(path, "vocals.wav"), vocals, sample_rate)

def augment_track(track, out_dir):
    # pitch_augment_track(track, out_dir)
    # slide_augment_track(track, out_dir)
    mix_augment_track(track, out_dir)

def mix_augment_track(track, out_dir):
    name = track.name
    vocals = track.targets["vocals"].audio
    acc = track.targets["accompaniment"].audio
    voxl = vocals.T[0]
    accl = acc.T[0]
    mixture = voxl + 0.8*accl
    path = os.path.join(out_dir, name+"_80")
    save_tracks(mixture, voxl, path)
    

def slide_augment_track(track, out_dir):
    name = track.name
    vocals = track.targets["vocals"].audio
    acc = track.targets["accompaniment"].audio
    acc = to_mono(acc)

    yvocals = rotate(vocals, 10)
    mixture = acc + yvocals
    path = os.path.join(out_dir, name+"_f10")
    save_tracks(mixture, yvocals, path)

    yvocals = rotate(vocals, 20)
    mixture = acc + yvocals
    path = os.path.join(out_dir, name+"_f20")
    save_tracks(mixture, yvocals, path)

def pitch_augment_track(track, out_dir):
    name = track.name
    vocals = track.targets["vocals"].audio
    acc = track.targets["accompaniment"].audio
    acc = to_mono(acc)

    yvocals = pitch_shift(vocals, 5)
    mixture = acc + yvocals
    path = os.path.join(out_dir, name+"_up")
    save_tracks(mixture, yvocals, path)

    yvocals = pitch_shift(vocals, -5)
    mixture = acc + yvocals
    path = os.path.join(out_dir, name+"_down")
    save_tracks(mixture, yvocals, path)
    

if __name__=="__main__":
    #args = docopt(__doc__)
    #root_dir = args["<musdb-root>"]
    root_dir = sys.argv[1]
    num_workers = hp.workers
    mus = musdb.DB(root_dir=root_dir, is_wav=True)
    tracks = mus.load_mus_tracks(subsets=["test"])
    out_dir = os.path.join(root_dir, "vup_test")
    with Manager() as manager:
        processes = []
        for track in tqdm(tracks):
            if len(processes) >= num_workers:
                for p in processes:
                    p.join()
                processes = []

            p = Process(target=augment_track, args=(track, out_dir))
            p.start()
            processes.append(p)
        if len(processes) > 0:
            for p in processes:
                p.join()
