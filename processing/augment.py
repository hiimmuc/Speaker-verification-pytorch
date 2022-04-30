import os
import random
import time
import wave

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from scipy import signal
from scipy.io import wavfile

from .audio_signal import compute_amplitude
from .wav_conversion import np_to_segment, segment_to_np, padding_np, normalize_audio_amp

# ====================================================Audio Augmentation utils==================================
## Time domain
    
def gain_target_amplitude(sound, target_dBFS=-10):
    if target_dBFS > 0:
        return sound
    change_in_dBFS = target_dBFS - sound.dBFS
    return sound.apply_gain(change_in_dBFS)

    
def random_augment_volume(signal, volume=6):
    states = ['higher', 'lower', 'unchange']
    state = np.random.choice(states, p=[0.5, 0.5, 0])
    
    if state == 'higher':
        gain = np.random.uniform(low=0, high=volume, size=None)
    elif state == 'lower':
        gain = np.random.uniform(low=-volume, high=0, size=None)
    else:
        gain = 0 # unchange speed
        
    return signal.apply_gain(gain)

def speed_change(sound, speed=1.0):
    # Manually override the frame_rate. This tells the computer how many
    # samples to play per second
    # slow_sound = speed_change(sound, 0.75)
    # fast_sound = speed_change(sound, 2.0)
    
    sound_with_altered_frame_rate = sound._spawn(sound.raw_data, overrides={
         "frame_rate": int(sound.frame_rate * speed)
      })
    # convert the sound with altered frame rate to a standard frame rate
    # so that regular playback programs will work right. They often only
    # know how to play audio at standard frame rate (like 44.1k)
    return sound_with_altered_frame_rate.set_frame_rate(sound.frame_rate)

def random_augment_speed(sound, low=0.95, high=1.05):
    states = ['faster', 'slower', 'unchange']
    state = np.random.choice(states, p=[0.5, 0.5, 0])
    
    if state == 'faster':
        speed = np.random.uniform(low=1.0, high=high, size=None)
    elif state == 'slower':
        speed = np.random.uniform(low=low, high=1.0, size=None)
    else:
        speed = 1.0 # unchange speed
        
    return speed_change(sound, speed)


def pitch_shift(sound, n_step=0.0, n_octave_bin=12, sr=8000):
    # shift the pitch up by half an octave (speed will increase proportionally)
    
    new_sample_rate = int(sound.frame_rate * (2.0 ** (n_step/n_octave_bin)))

    # keep the same samples but tell the computer they ought to be played at the 
    # new, higher sample rate. This file sounds like a chipmunk but has a weird sample rate.
    hipitch_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate})

    # now we just convert it to a common sample rate (44.1k - standard audio CD) to 
    # make sure it works in regular audio players. Other than potentially losing audio quality (if
    # you set it too low - 44.1k is plenty) this should now noticeable change how the audio sounds.
    hipitch_sound = hipitch_sound.set_frame_rate(sr)    
    return hipitch_sound

def random_augment_pitch_shift(x, nstep_low=-0.5, n_step_high=0.5):
    states = ['higher', 'lower', 'unchange']
    state = np.random.choice(states, p=[0.5, 0.5, 0])
    
    if state == 'higher':
        n_step = np.random.uniform(low=0, high=n_step_high, size=None)
    elif state == 'lower':
        n_step = np.random.uniform(low=nstep_low, high=0, size=None)
    else:
        n_step = 0 # unchange speed
        
    return pitch_shift(x, n_step)
    

def random_drop_chunk(sound, lengths,         
                      drop_length_low=100,
                      drop_length_high=1000,
                      drop_count_low=1,
                      drop_count_high=10,
                      drop_start=0,
                      drop_end=None,
                      drop_prob=1,
                      noise_factor=0.0):
    """This fucntion drops portions of the input signal.
    Using `DropChunk` as an augmentation strategy helps a models learn to rely
    on all parts of the signal, since it can't expect a given part to be
    present.
    Arguments
    ---------
    drop_length_low : int
        The low end of lengths for which to set the
        signal to zero, in samples.
    drop_length_high : int
        The high end of lengths for which to set the
        signal to zero, in samples.
    drop_count_low : int
        The low end of number of times that the signal
        can be dropped to zero.
    drop_count_high : int
        The high end of number of times that the signal
        can be dropped to zero.
    drop_start : int
        The first index for which dropping will be allowed.
    drop_end : int
        The last index for which dropping will be allowed.
    drop_prob : float
        The probability that the batch of signals will
        have a portion dropped. By default, every batch
        has portions dropped.
    noise_factor : float
        The factor relative to average amplitude of an utterance
        to use for scaling the white noise inserted. 1 keeps
        the average amplitude the same, while 0 inserts all 0's.
    waveforms : ndarray
        Shape should be `[batch, time]` or `[batch, time, channels]`.
    lengths : ndarray
        Shape should be a single dimension, `[batch]`.
    Returns
    -------
    ndarray of shape `[batch, time]` or
        `[batch, time, channels]`
    """
    # Validate low < high
    if drop_length_low > drop_length_high:
        raise ValueError("Low limit must not be more than high limit")
    if drop_count_low > drop_count_high:
        raise ValueError("Low limit must not be more than high limit")

    # Make sure the length doesn't exceed end - start
    if drop_end is not None and drop_end >= 0:
        if drop_start > drop_end:
            raise ValueError("Low limit must not be more than high limit")

        drop_range = drop_end - drop_start
        drop_length_low = min(drop_length_low, drop_range)
        drop_length_high = min(drop_length_high, drop_range)
    
    batch_size = 1
    sound = np.expand_dims(sound, 0)

    lengths = (lengths * sound.shape[0])
    dropped_waveform = np.copy(sound)
    
    if np.random.rand(1) > drop_prob:
        return dropped_waveform
    
    clean_amplitude = compute_amplitude(sound)

    # Pick a number of times to drop
    drop_times = np.random.randint(
        low=drop_count_low,
        high=drop_count_high + 1,
        size=(batch_size,),
    )

    # Iterate batch to set mask
    for i in range(batch_size):
        if drop_times[i] == 0:
            continue

        # Pick lengths
        length = np.random.randint(
            low=drop_length_low,
            high=drop_length_high + 1,
            size=(drop_times[i],),
        )

        # Compute range of starting locations
        start_min = drop_start
        if start_min < 0:
            start_min += lengths[i]
        start_max = drop_end
        if start_max is None:
            start_max = lengths[i]
        if start_max < 0:
            start_max += lengths[i]
        start_max = max(0, start_max - length.max())

        # Pick starting locations
        start = np.random.randint(
            low=start_min, high=start_max + 1, size=(drop_times[i],),
        )

        end = start + length

        # Update waveform
        if not noise_factor:
            for j in range(drop_times[i]):
                dropped_waveform[i, start[j] : end[j]] = 0.0
        else:
            # Uniform distribution of -2 to +2 * avg amplitude should
            # preserve the average for normalization
            noise_max = 2 * clean_amplitude[i] * noise_factor
            for j in range(drop_times[i]):
                # zero-center the noise distribution
                noise_vec = np.random.rand(length[j])
                noise_vec = 2 * noise_max * noise_vec - noise_max
                dropped_waveform[i, start[j] : end[j]] = noise_vec
    return dropped_waveform.squeeze().to_numpy()