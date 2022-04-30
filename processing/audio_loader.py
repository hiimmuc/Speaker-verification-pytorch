import glob
import os
import random
import sys
import time
import wave

import numpy as np
import soundfile as sf
from pydub import AudioSegment
from scipy import signal
from scipy.io import wavfile

from .augment import (gain_target_amplitude, random_augment_pitch_shift,
                      random_augment_speed, random_augment_volume)
from .wav_conversion import np_to_segment, segment_to_np


def random_augment_audio(audio_seg, p=[0, 0.25, 0.25, 0.25, 0.25]):
    aug_types = ['all', 'speed', 'pitch', 'volume', 'none']
    aug_type = np.random.choice(aug_types, p=p)

    if aug_type == 'all':
        # drop chunk
        '''Upcoming'''
        # augment speed
        audio_seg = random_augment_speed(audio_seg, 0.95, 1.05)
        # pitch shift
        audio_seg = random_augment_pitch_shift(audio_seg, -0.5, 0.5)
        # augment volume
        audio_seg = random_augment_volume(audio_seg, volume=6)
    elif aug_type == 'speed':
        audio_seg = random_augment_speed(audio_seg, 0.95, 1.05)
    elif aug_type == 'pitch':
        audio_seg = random_augment_pitch_shift(audio_seg, -0.5, 0.5)
    elif aug_type == 'volume':
        audio_seg = random_augment_volume(audio_seg, volume=6)
    else:
        pass

    return audio_seg

# ================================================Utils============================================


def loadWAV(audio_source, max_frames,
            evalmode=True, num_eval=10, sample_rate=16000,
            augment=False, augment_chain=None, target_db=None,
            read_mode='pydub', **kwargs):
    '''Load audio form .wav file and return as the np array

    Args:
        audio_source (str or numpy array): [description]
        max_frames ([int]): max number of frames to get, -1 for entire audio
        evalmode (bool, optional): [description]. Defaults to True.
        num_eval (int, optional): [description]. Defaults to 10.
        sr ([type], optional): [description]. Defaults to None.
        augment([bool]): decide wether apply augment on loading aduio(time domain)
        augment_chain(list, str): chain of augment to apply(if augment == True). available: env_corrupt time_domain spec_domain
    Returns:
        ([ndarray]): audio_array
    '''
    if isinstance(audio_source, str):
        if read_mode == 'sf':
            audio, sample_rate = sf.read(audio_source)
        else:
            audio_seg = AudioSegment.from_file(audio_source)
            sr = int(audio_seg.frame_rate)

            assert sample_rate == sr, f"Sample rate is not same as desired value {sample_rate} and {sr}"
            sample_rate = sr

            if augment and ('time_domain' in augment_chain):
                audio_seg = random_augment_audio(audio_seg)
            if target_db is not None:
                audio_seg = gain_target_amplitude(audio_seg, target_db)

            # convert to numpy with soundfile mormalize format
            audio = segment_to_np(audio_seg, normalize=True)

    elif isinstance(audio_source, np.ndarray):
        audio = audio_source
    else:
        raise "Invalid format of audio source, available: str, ndarray"

    audiosize = audio.shape[0]

    # Maximum audio length counted in frames
    # hoplength is 160, winlength is 400 -> total length  = winlength- hop_length + max_frames * hop_length
    # get the winlength 25ms, hop 10ms
    hoplength = 10e-3 * sample_rate
    winlength = 25e-3 * sample_rate

    if max_frames > 0:
        max_audio = int(max_frames * hoplength + (winlength - hoplength))

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]

        if evalmode:
            # get num_eval of audio and stack together
            startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            # get randomly initial index of frames, not always from 0
            startframe = np.array(
                [np.int64(random.random() * (audiosize - max_audio))])

        feats = []
        if evalmode and num_eval == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = np.stack(feats, axis=0).astype(np.float)

        return feat
    else:
        return audio

# Environment corruption


class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames, sample_rate=8000, target_db=None):
        self.sr = sample_rate
        self.target_db = target_db

        hop_length = 10e-3 * self.sr
        win_length = 25e-3 * self.sr

        self.max_frames = max_frames
        self.max_audio = int(max_frames * hop_length + (win_length - hop_length))

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {
            'noise': [0, 12],
            'speech': [2, 12],
            'music': [0, 12],
            'noise_vad': [0, 12]
        }
        self.num_noise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1], 'noise_vad': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))

        # custom noise use additive noise latter
        noise_vad_files = glob.glob("dataset/noise_vad/noise_vad/*/*.wav")
        for noise_file in noise_vad_files:
            if noise_file.split('/')[-3] not in self.noiselist:
                self.noiselist[noise_file.split('/')[-3]] = []
            self.noiselist[noise_file.split('/')[-3]].append(noise_file)
        print(len(self.rir_files), len(augment_files))

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        num_noise = self.num_noise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat],
                                  random.randint(num_noise[0], num_noise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False, sample_rate=self.sr, target_db=self.target_db)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],
                                       self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        aug_audio = np.sum(np.concatenate(noises, axis=0), axis=0, keepdims=True) + audio
        return aug_audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)
        rir = loadWAV(rir_file, max_frames=-1, evalmode=False, sample_rate=self.sr, target_db=self.target_db)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        aug_audio = signal.convolve(audio, rir, mode='full')[:, :self.max_audio]
        return aug_audio


"""
Note for 
with open(audio_path,'rb') as rf:
    rf.read()
    
Positions  Typical Value Description
1 - 4      "RIFF"        Marks the file as a RIFF multimedia file.
                         Characters are each 1 byte long.
5 - 8      (integer)     The overall file size in bytes (32-bit integer)
                         minus 8 bytes. Typically, you'd fill this in after
                         file creation is complete.
9 - 12     "WAVE"        RIFF file format header. For our purposes, it
                         always equals "WAVE".
13-16      "fmt "        Format sub-chunk marker. Includes trailing null.
17-20      16            Length of the rest of the format sub-chunk below.
21-22      1             Audio format code, a 2 byte (16 bit) integer. 
                         1 = PCM (pulse code modulation).
23-24      2             Number of channels as a 2 byte (16 bit) integer.
                         1 = mono, 2 = stereo, etc.
25-28      44100         Sample rate as a 4 byte (32 bit) integer. Common
                         values are 44100 (CD), 48000 (DAT). Sample rate =
                         number of samples per second, or Hertz.
29-32      176400        (SampleRate * BitsPerSample * Channels) / 8
                         This is the Byte rate.
33-34      4             (BitsPerSample * Channels) / 8
                         1 = 8 bit mono, 2 = 8 bit stereo or 16 bit mono, 4
                         = 16 bit stereo.
35-36      16            Bits per sample. 
37-40      "data"        Data sub-chunk header. Marks the beginning of the
                         raw data section.
41-44      (integer)     The number of bytes of the data section below this
                         point. Also equal to (#ofSamples * #ofChannels *
                         BitsPerSample) / 8
45+                      The raw audio data. 
"""
