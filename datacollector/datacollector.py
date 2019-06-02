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
win_offset = sample_rate*30
window = sample_rate*15
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
    return np.sum(x[0,win_offset:win_offset+window]*y[0,win_offset:win_offset+window])

def correlate_signals(mixture_path, vocal_path, outpath):
    mwav = load_wav(mixture_path)
    mwav = slice_onset(mwav)
    vwav = load_wav(vocal_path)
    vwav = slice_onset(vwav)
    fine_win = sample_rate//10
    
    offsets_coarse = [i for i in range(0, scan_range, 10)]
    corrs = np.array([compare_signals(mwav[:,i:], vwav) 
                      for i in tqdm(offsets_coarse)])
    sync_point = offsets_coarse[np.where(corrs == np.max(corrs))[0][0]]
    fine_start = max(sync_point-fine_win, 0)
    fine_end = min(sync_point+fine_win, mwav.shape[-1])
    offsets_fine = [i for i in range(fine_start, fine_end)]
    corrs = np.array([compare_signals(mwav[:,i:], vwav)
                      for i in tqdm(offsets_fine)])
    sync_point = offsets_fine[np.where(corrs == np.max(corrs))[0][0]]
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

def synchronize():
    stems = os.listdir(unprocessed_dir)
    for stem in stems:
        outpath = os.path.join(output_dir, stem)
        os.makedirs(outpath, exist_ok=True)
        mixture_path = os.path.join(unprocessed_dir, stem, "mixture.wav")
        vocal_path = os.path.join(unprocessed_dir, stem, "vocals.wav")
        try:
            correlate_signals(mixture_path, vocal_path, outpath)
        except Exception as e:
            print(f"Error encountered while processing {stem}")

def setup():
    os.makedirs(unprocessed_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    acapellas = os.listdir(acapellas_dir)
    return acapellas

if __name__=='__main__':
    acapellas = setup()
    download_mixtures(acapellas)
    synchronize()
