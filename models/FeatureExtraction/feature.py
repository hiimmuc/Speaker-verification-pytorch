import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio import features
from utils import PreEmphasis
import librosa
import numpy as np

def librosa_mfcc(x):
    x = np.squeeze(x)
    S = librosa.feature.melspectrogram(x, sr=8000, 
                                       n_mels=80, n_fft=512,
                                       hop_length=80, win_length=200, 
                                       window='hamming', 
                                       fmin=10.0, fmax=4000,
                                       center=True, pad_mode='reflect', 
                                       power=2.0, norm='slaney')
    S = librosa.power_to_db(S)
    mfcc = librosa.feature.mfcc(S=S, sr=8000, n_mfcc=80, norm='ortho')
    return torch.FloatTensor(np.expand_dims(mfcc, 0))

def librosa_mel(x):
    x = np.squeeze(x)
    mel = librosa.feature.melspectrogram(x, sr=8000, 
                                       n_mels=80, n_fft=512,
                                       hop_length=80, win_length=200, 
                                       window='hamming', 
                                       fmin=10.0, fmax=4000,
                                       center=True, pad_mode='reflect', 
                                       power=2.0, norm='slaney')
    return torch.FloatTensor(np.expand_dims(mel, 0))

def mfcc(lib='nnaudio', sr = 8000, 
         n_fft=512, win_length=200, n_mfcc=80, 
         n_mels=80,hop_length=80, window='hamming', 
         fmin=10.0, fmax=4000, pre_emphasis=True, **kwargs):
    
    if lib.lower() == 'librosa':
        return librosa_mfcc

    if lib.lower() == 'nnaudio':
        feat = features.mel.MFCC(sr=sr, n_fft=n_fft, 
                                 win_length=win_length, 
                                 n_mfcc=n_mfcc, n_mels=n_mels, 
                                 hop_length=hop_length, 
                                 window=window, 
                                 fmin=fmin, fmax=fmax, 
                                 verbose=False)
    elif lib.lower() == 'torchaudio':
        window_fn = torch.hamming_window if window == 'hamming' else torch.hann_window
        feat = torchaudio.transforms.MFCC(n_mfcc= n_mfcc, 
                                          sample_rate=sr, 
                                          melkwargs={"n_fft": n_fft, 
                                                     "hop_length": hop_length, 
                                                     "win_length" :win_length,
                                                     'f_min': fmin, 'f_max': fmax, 
                                                     'window_fn': window_fn, 
                                                     "power": 2, 
                                                     'n_mels': n_mels,
                                                     'norm':'slaney',
                                                     'mel_scale': 'slaney'})
    return torch.nn.Sequential(PreEmphasis(), feat) if pre_emphasis else feat

    
def melspectrogram(lib = 'nnaudio', 
                   sr=8000, n_fft=512, 
                   win_length=200, n_mels=80, 
                   hop_length=80, window='hamming',
                   fmin=0.0, fmax=None,
                   verbose=False, pre_emphasis=True, **kwargs):
    
    if lib.lower() == 'librosa':
        return librosa_mel
    
    if lib.lower() == 'nnaudio':
        feat = features.mel.MelSpectrogram(sr=sr, n_fft=n_fft, 
                                           win_length=win_length, 
                                           n_mels=n_mels, 
                                           hop_length=hop_length, 
                                           window=window, 
                                           fmin=fmin, fmax=fmax, 
                                           verbose=False)
    elif lib.lower() == 'torchaudio':
        window_fn = torch.hamming_window if window == 'hamming' else torch.hann_window
        feat = torchaudio.transforms.MelSpectrogram(n_mels= n_mels, 
                                                    sample_rate=sr, n_fft=n_fft,
                                                    win_length=win_length,
                                                    hop_length=hop_length,
                                                    f_min=fmin, f_max=fmax,
                                                    window_fn=window_fn,
                                                    norm='slaney',
                                                    mel_scale='slaney')
    return torch.nn.Sequential(PreEmphasis(),feat) if pre_emphasis else feat
                                                    

    