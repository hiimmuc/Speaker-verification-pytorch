from asyncio import subprocess
import os
import wave
import numpy as np
from pydub import AudioSegment
from scipy import signal


def np_to_segment(np_array, sr=8000):
    # ignore normalized audio
    assert np_array.dtype.itemsize == 2, "Audio is normalized!"

    audio_segment = AudioSegment(
        np_array.tobytes(),
        frame_rate=sr,
        sample_width=np_array.dtype.itemsize,
        channels=1)
    return audio_segment


def segment_to_np(segment, normalize=False, dtype=np.float64):
    audio_array = segment.get_array_of_samples()
    audio_np = np.array(audio_array)
    if normalize:
        audio_np = normalize_audio_amp(audio_np)
    return audio_np.astype(dtype)


def padding_np(audio, length_target):
    shortage = length_target - audio.shape[0]
    audio = np.pad(audio, (0, shortage), 'wrap')
    return audio


def normalize_audio_amp(signal):
    try:
        intinfo = np.iinfo(signal.dtype)
        return signal / max(intinfo.max, -intinfo.min)

    except ValueError:  # array is not integer dtype
        return signal / max(signal.max(), -signal.min())


def convert_audio_pydub(src, ext='wav', sample_rate=8000, channels=1, codec='pcm_s16le ', dst=None):
    """Convert audio format and samplerate to target"""
    try:
        org_format = src.split('.')[-1].strip()
        if ext != org_format:
            audio = AudioSegment.from_file(src)
            # export file as new format
            src = src.replace(org_format, ext)
            audio.export(src, format=ext)
    except Exception as e:
        raise e

    try:
        sound = AudioSegment.from_file(src, format='wav')
        sound = sound.set_frame_rate(sample_rate)
        sound = sound.set_channels(channels)

        dst = src if not dst else dst

        sound.export(dst, format='wav')
    except Exception as e:
        raise e

    return dst


def convert_audio_shell(src, ext='wav', sample_rate=8000, channels=1, codec='pcm_s16le ', dst=None):
    dst = src if not dst else dst
    old_ext = str(src).split('.')[-1]
    dst = str(dst).replace(old_ext, ext)
    cmd = f'ffmpeg -i {src} -acodec {codec} -ar {sample_rate} -ac {channels} {dst}'
    subprocess.call(cmd, shell=True)
    return dst


if __name__ == '__main__':
    pass
