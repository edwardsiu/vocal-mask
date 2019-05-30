import librosa
import librosa.onset
import librosa.output
import subprocess
import os
import numpy as np
import re
import sys
import youtube_dl
from multiprocessing import Manager, Process
from tqdm import tqdm
import time

acapellas_dir = "acapellas"
unprocessed_dir = "unprocessed"
output_dir = "stems"
sample_rate = 44100
window = sample_rate*10
scan_range = sample_rate*30

def get_file_name(path):
    match = re.match('^[\w&\.-]*', path[:-4])
    fname = match.group(0).strip('_')
    return fname

def mp3_to_wav(inpath, outpath):
    subprocess.call(['ffmpeg','-i',inpath,outpath])

def load_wav(path):
    wav = librosa.load(path, sr=sample_rate, mono=False)[0]
    return scale_signal(wav)

def scale_signal(wav):
    return wav/np.max(np.abs(wav))

def slice_onset(wav):
    onsets = librosa.onset.onset_detect(wav[0], sr=sample_rate, 
                units='samples', backtrack=True)
    return wav[:,onsets[0]:]

def compare_signals(x, y):
    return np.sum(x[0,:window]*y[0,:window])

def correlate_signals(mixture_path, vocal_path, outpath):
    mwav = load_wav(mixture_path)
    mwav = slice_onset(mwav)
    vwav = load_wav(vocal_path)
    vwav = slice_onset(vwav)
    
    corrs = np.array([compare_signals(mwav[:,i:], vwav) 
                      for i in tqdm(range(scan_range))])
    sync_point = np.where(corrs == np.max(corrs))[0][0]
    mwav = mwav[:,sync_point:]
    if mwav.shape[1] > vwav.shape[1]:
        mwav = mwav[:,:vwav.shape[1]]
    else:
        vwav = vwav[:,:mwav.shape[1]]

    mixture_path = os.path.join(outpath, 'mixture.wav')
    vocal_path = os.path.join(outpath, 'vocals.wav')
    librosa.output.write_wav(mixture_path, mwav, sr=sample_rate, norm=True)
    librosa.output.write_wav(vocal_path, vwav, sr=sample_rate, norm=True)

def download_mixture(search_key, outpath):
    opts = {
        'format': 'bestaudio/best',
        'outtmpl': outpath,
        'prefer_ffmpeg': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav'
        }]
    }
    with youtube_dl.YoutubeDL(opts) as ydl:
        ydl.download([f'ytsearch:{search_key}'])

def download_mixtures(acapellas):
    for f in acapellas:
        fname = get_file_name(f)
        search_key = fname.replace('_',' ')
        outpath = os.path.join(unprocessed_dir, fname)
        os.makedirs(outpath, exist_ok=True)
        vocal_path = os.path.join(outpath, "vocals.wav")
        mixture_path = os.path.join(outpath, "mixture.wav")
        mp3_to_wav(os.path.join(acapellas_dir, f), vocal_path)
        download_mixture(search_key, mixture_path)
        time.sleep(0.1)

def process_multi(num_workers):
    stems = os.listdir(unprocessed_dir)
    with Manager() as manager:
        processes = []
        for stem in tqdm(stems):
            if len(processes) >= num_workers:
                for p in processes:
                    p.join()
                processes = []
            outpath = os.path.join(output_dir, stem)
            os.makedirs(outpath, exist_ok=True)
            mixture_path = os.path.join(unprocessed_dir, stem, "mixture.wav")
            vocal_path = os.path.join(unprocessed_dir, stem, "vocals.wav")
            p = Process(target=correlate_signals, args=(mixture_path, vocal_path, outpath))
            p.start()
            processes.append(p)
        if len(processes) > 0:
            for p in processes:
                p.join()

def synchronize():
    stems = os.listdir(unprocessed_dir)
    for stem in stems:
        outpath = os.path.join(output_dir, stem)
        os.makedirs(outpath, exist_ok=True)
        mixture_path = os.path.join(unprocessed_dir, stem, "mixture.wav")
        vocal_path = os.path.join(unprocessed_dir, stem, "vocals.wav")
        correlate_signals(mixture_path, vocal_path, outpath)

def setup():
    os.makedirs(unprocessed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    acapellas = os.listdir(acapellas_dir)
    return acapellas

if __name__=='__main__':
    acapellas = setup()
    download_mixtures(acapellas)
    synchronize()
