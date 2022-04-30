import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from nnAudio import features
from utils import PreEmphasis


def mfcc(lib='nnaudio', sr = 8000, 
         n_fft=512, win_length=200, n_mfcc=80, 
         n_mels=80,hop_length=80, window='hamming', 
         fmin=10.0, fmax=4000, **kwargs):

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
    return torch.nn.Sequential(PreEmphasis(), feat)

    
def melspectrogram(lib = 'nnaudio', 
                   sr=8000, n_fft=512, 
                   win_length=200, n_mels=80, 
                   hop_length=80, window='hamming',
                   fmin=0.0, fmax=None,
                   verbose=False, **kwargs):
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
                                                    f_min=f_min, f_max=f_max,
                                                    window_fn=window_fn,
                                                    norm='slaney',
                                                    mel_scale='slaney')
    return torch.nn.Sequential(PreEmphasis(), feat)
                                                    

    