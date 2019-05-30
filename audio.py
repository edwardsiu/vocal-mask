import librosa
import librosa.effects
import librosa.feature
import librosa.filters
import librosa.output
import math
import numpy as np
from scipy import signal
from hparams import hparams
from scipy.io import wavfile


def load_wav(path, mono=True, offset=0.0, duration=None):
    wav = librosa.load(path, sr=hparams.sample_rate, mono=mono, offset=offset, duration=duration)[0]
    return wav

def save_wav(wav, path, sr=None):
    if not sr:
        sr = hparams.sample_rate
    librosa.output.write_wav(path, wav, sr=sr)

def get_wav_slices(wav, window, stride):
    N = len(wav)
    return [(i,i+window) for i in range(0, N-window, stride)]

def show_spec(spec, y_axis='mel'):
    librosa.display.specshow(spec, sr=hparams.sample_rate, y_axis=y_axis, x_axis='time', hop_length=hparams.hop_size)

def preemphasis(x):
    from nnmnkwii.preprocessing import preemphasis
    return preemphasis(x, hparams.preemphasis)


def inv_preemphasis(x):
    from nnmnkwii.preprocessing import inv_preemphasis
    return inv_preemphasis(x, hparams.preemphasis)

def hpss_decompose(S):
    H, P = librosa.decompose.hpss(S, margin=2.0)
    R = S - H - P
    return H, P, R

def make_vocal_mask(Smix, Starget):
    if hparams.mask_type == 'IBM':
        ibm_mask = IBM(Smix, Starget)
        S = mel_weight(Starget, power=hparams.power["vox"])
        S = scale_spec(S, per_channel=hparams.per_channel_norm["vox"])
        thr_mask = (S >= hparams.threshold)
        return ibm_mask * thr_mask
    else:
        S = mel_weight(Starget, power=hparams.power["vox"])
        S = scale_spec(S, per_channel=hparams.per_channel_norm["vox"])
        return (S >= hparams.threshold)

def IBM(Smix, Starget):
    eps = np.finfo(np.float).eps
    Starget = np.abs(Starget)**hparams.IBM['alpha']
    Smix = np.abs(Smix)**hparams.IBM['alpha']
    mask = np.divide(Starget, (eps + Smix))
    
    mask[np.where(mask >= hparams.IBM['theta'])] = 1
    mask[np.where(mask < hparams.IBM['theta'])] = 0
    mask[np.where(Starget < hparams.IBM['theta2'])] = 0
    return mask.astype(np.bool)

def stft(y, preemp=False):
    if preemp:
        y = preemphasis(y)
    return librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)

def scaled_mel_weight(S, power, per_channel):
    M = mel_weight(S, power)
    return scale_spec(M, per_channel=per_channel)

def mel_weight(S, power):
    global _mel_freqs
    if _mel_freqs is None:
        _mel_freqs = librosa.mel_frequencies(S.shape[0], fmin=hparams.fmin)
    S = librosa.perceptual_weighting(np.abs(S)**power, _mel_freqs, ref=hparams.ref_level_db)
    S = _normalize(S - hparams.ref_level_db)
    return S

def scale_spec(S, per_channel=True):
    if per_channel:
        S = np.clip(S - np.nanmean(S, axis=1, keepdims=True), 0, 1)
        m = np.max(S, axis=1, keepdims=True)
    else:
        S = np.clip(S - np.nanmean(S), 0, 1)
        m = np.max(S, axis=1, keepdims=True)
    return S/(1e-8 + m)

def batch_scale_specs(S, per_channel=True):
    if per_channel:
        S = np.clip(S - np.nanmean(S, axis=3, keepdims=True), 0, 1)
        m = np.max(S, axis=3, keepdims=True)
    else:
        S = np.clip(S - np.nanmean(S, axis=(1,2,3), keepdims=True), 0, 1)
        m = np.max(S, axis=3, keepdims=True)
    return S/(1e-8 + m)

def spectrogram(y, power):
    global _mel_freqs
    stftS = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)
    if hparams.use_preemphasis:
        y = preemphasis(y)
    S = librosa.stft(y, n_fft=hparams.fft_size, hop_length=hparams.hop_size)
    if _mel_freqs is None:
        _mel_freqs = librosa.mel_frequencies(S.shape[0], fmin=hparams.fmin)
    _S = librosa.perceptual_weighting(np.abs(S)**power, _mel_freqs, ref=hparams.ref_level_db)
    return _normalize(_S - hparams.ref_level_db), stftS

def inv_spectrogram(S):
    y = librosa.istft(S, hop_length=hparams.hop_size)
    return y
    

# Conversions:
_mel_freqs = None

def _amp_to_db(x):
    min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
    return 20 * np.log10(np.maximum(min_level, x))


def _db_to_amp(x):
    return np.power(10.0, x * 0.05)


def _normalize(S):
    return np.clip((S - hparams.min_level_db) / -hparams.min_level_db, 0, 1)


def _denormalize(S):
    return (np.clip(S, 0, 1) * -hparams.min_level_db) + hparams.min_level_db
