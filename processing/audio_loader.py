import glob
import os
import random

import numpy as np
import torchaudio
import soundfile as sf
from pydub import AudioSegment

from scipy import signal

from .augment import (random_augment_speed, random_augment_pitch_shift,
                      random_augment_volume, gain_target_amplitude)
from .wav_conversion import segment_to_np, np_to_segment, normalize_audio_amp
from pathlib import Path


def random_augment_audio(audio_seg, setting):
    aug_types = ['speed', 'pitch', 'volume', 'none']
    if len(setting['proportion']) < len(aug_types):
        p = setting['proportion'] + \
            [1 - sum(setting['proportion'])
             ]  # [0.25, 0.25, 0.25] -> [0.25, 0.25, 0.25, 0.25]

    aug_type = np.random.choice(aug_types, p=p)

    if setting['combined']:
        # drop chunk
        '''Upcoming'''
        # augment speed
        audio_seg = random_augment_speed(
            audio_seg, setting['speed'][0], setting['speed'][1])
        # pitch shift
        audio_seg = random_augment_pitch_shift(
            audio_seg, setting['pitch'][0], setting['pitch'][1])
        # augment volume
        audio_seg = random_augment_volume(audio_seg, volume=setting['volume'])
    elif aug_type == 'speed':
        audio_seg = random_augment_speed(
            audio_seg, setting['speed'][0], setting['speed'][1])
    elif aug_type == 'pitch':
        audio_seg = random_augment_pitch_shift(
            audio_seg, setting['pitch'][0], setting['pitch'][1])
    elif aug_type == 'volume':
        audio_seg = random_augment_volume(audio_seg, volume=setting['volume'])
    else:
        pass

    return audio_seg

# ================================================Utils============================================


def loadWAV(audio_source, audio_spec,
            evalmode=True, num_eval=10,
            augment=False, augment_options=None, target_db=None,
            read_mode='pydub', random_chunk=True, load_all=False, dtype=np.float32, ** kwargs):
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
    set_sample_rate = audio_spec['sample_rate']

    if isinstance(audio_source, str):
        audio_source = str(Path(audio_source))  # to compatible with most os
        if read_mode == 'sf':
            audio, sr = sf.read(audio_source)
        else:
            # read audio using Audio Segments
            audio_seg = AudioSegment.from_file(audio_source)
            sr = int(audio_seg.frame_rate)

            assert set_sample_rate == sr, f"Sample rate is not same as desired value {set_sample_rate} and {sr}"
            
            # normalize rms value of DB to target_db if specified
            if target_db is not None:
                audio_seg = gain_target_amplitude(audio_seg, target_db)
            
            # perform time donmain augmentation
            if augment and ('time_domain' in augment_options['augment_chain']):
                audio_seg = random_augment_audio(
                    audio_seg, augment_options['augment_time_domain'])

            # convert to numpy with soundfile normalize format
            audio = segment_to_np(audio_seg, normalize=True, dtype=dtype)

    elif isinstance(audio_source, np.ndarray):
        audio = normalize_audio_amp(audio_source)
    else:
        raise "Invalid format of audio source, available: str, ndarray"

    
    if load_all:
        return np.expand_dims(audio, 0)
    else:
        audiosize = audio.shape[0]
        
        # Maximum audio length counted in frames
        # winlength 25ms, and hop 10ms with sr = 8000 -> hoplen = 80 winlen = 200 
        # total length  = (winlength- hop_length) + max_frames * hop_length

        n_hop_frames = int(audio_spec['hop_len'] * set_sample_rate)
        n_win_frames = int(audio_spec['win_len'] * set_sample_rate)
        max_audio = int(audio_spec['sentence_len'] * set_sample_rate) # max_audio = int(max_frames * n_hop_frames + n_overlap_frames)

        n_overlap_frames = n_win_frames - n_hop_frames
        max_frames = round((max_audio - n_overlap_frames) / n_hop_frames)
        assert max_frames > 0, "invalid size of frame"

        if audiosize <= max_audio:
            shortage = max_audio - audiosize + 1
            audio = np.pad(audio, (0, shortage), 'wrap')
            audiosize = audio.shape[0]
        
        ## get start index
        if evalmode:
            # get num_eval of audio and stack together
            startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
        else:
            # get randomly initial index of frames, not always from 0
            if random_chunk:
                startframe = np.array(
                    [np.int64(random.random() * (audiosize - max_audio))])
            else:
                startframe = np.array([0])
                
        feats = []
        if evalmode and num_eval == 0:
            feats.append(audio)
        else:
            for asf in startframe:
                feats.append(audio[int(asf):int(asf) + max_audio])

        feat = np.stack(feats, axis=0).astype(dtype)

        return feat


# Environment corruption


class AugmentWAV(object):
    def __init__(self, augment_options, audio_spec, target_db=None):
        self.target_db = target_db

        self.augment_chain = augment_options['augment_chain']
        self.musan_path = augment_options['augment_paths']['musan']
        self.noise_vad_path = augment_options['augment_paths']['noise_vad']
        self.rirs_path = augment_options['augment_paths']['rirs']

        self.audio_spec = audio_spec
        self.sr = int(audio_spec['sample_rate'])
        self.n_hop_frames = int(audio_spec['hop_len'] * self.sr)
        self.n_win_frames = int(audio_spec['win_len'] * self.sr)
        self.max_audio = int(audio_spec['sentence_len'] * self.sr)

        n_overlap_frames = self.n_win_frames - self.n_hop_frames
        assert n_overlap_frames > 0, 'invalid audio spec'

        self.max_frames = round(
            (self.max_audio - n_overlap_frames) / self.n_hop_frames)

        self.noisesnr = dict(augment_options['noise_snr'])

        self.num_noise = dict(augment_options['noise_samples'])

        self.noiselist = {}

        print("Augment set information...")
        # dataset/augment_data/musan_split/ + noise/free-sound/noise-free-sound-0000.wav
        # dataset/augment_data/musan_split/ + music/fma/music-fma-0003.wav
        musan_noise_files = glob.glob(
            os.path.join(self.musan_path, '*/*/*/*.wav'))

        print(f"Using {len(musan_noise_files)} files of MUSAN noise")

        for file in musan_noise_files:
            assert file.split('/')[-4] in ['noise', 'speech', 'music']
            self.noiselist.setdefault(file.split('/')[-4], []).append(file)

        # custom noise use additive noise latter
        noise_vad_files = glob.glob(
            os.path.join(self.noise_vad_path, '*/*.wav'))
        print(f"Using {len(noise_vad_files)} files of VAD noise")
        for file in noise_vad_files:
            self.noiselist.setdefault('noise_vad', []).append(file)

        # noise from rirs noise
        rir_noise_files = glob.glob(os.path.join(f'{self.rirs_path}/pointsource_noises/', '*.wav')) + glob.glob(
            os.path.join(f'{self.rirs_path}/real_rirs_isotropic_noises/', '*.wav'))
        print(f"Using {len(rir_noise_files)} files of rirs noise")
        for file in rir_noise_files:
            self.noiselist.setdefault('noise_rirs', []).append(file)

        # RIRS_NOISES/simulated_rirs/ + smallroom/Room001/Room001-00001.wav
        self.reverberation_files = glob.glob(
            os.path.join(f'{self.rirs_path}/simulated_rirs/', '*/*/*.wav'))
        print(f"Using {len(self.reverberation_files)} files of rirs reverb")

    def additive_noise(self, noisecat, audio):

        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        num_noise = self.num_noise[noisecat]

        noiselist = random.sample(self.noiselist[noisecat],
                                  random.randint(num_noise[0], num_noise[1]))
        noises = []
        for noise in noiselist:
            noiseaudio = loadWAV(noise, audio_spec=self.audio_spec, evalmode=False,
                                 sample_rate=self.sr, target_db=self.target_db)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],
                                       self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) * noiseaudio)
        aug_audio = np.sum(np.concatenate(noises, axis=0),
                           axis=0, keepdims=True) + audio
        return aug_audio

    def reverberate(self, audio):
        rir_file = random.choice(self.reverberation_files)
        rir = loadWAV(rir_file, self.audio_spec, evalmode=False,
                      sample_rate=self.sr, target_db=self.target_db, load_all=True).astype(np.float32)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        aug_audio = signal.convolve(audio, rir, mode='full')[
            :, :self.max_audio]
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
