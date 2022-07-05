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

def segment_to_np(segment, normalize=False):
    audio_array = segment.get_array_of_samples()
    audio_np = np.array(audio_array)
    if normalize:
        audio_np = normalize_audio_amp(audio_np)
    return audio_np.astype(np.float64)

def padding_np(audio, length_target):
    shortage = length_target - audio.shape[0]
    audio = np.pad(audio, (0, shortage), 'wrap')
    return audio

def normalize_audio_amp(signal):
    try:
        intinfo = np.iinfo(signal.dtype)
        return signal / max( intinfo.max, -intinfo.min )

    except ValueError: # array is not integer dtype
        return signal / max( signal.max(), -signal.min())

def convert_audio(audio_path, new_format='wav', freq=8000, out_path=None):
    """Convert audio format and samplerate to target"""
    try:
        org_format = audio_path.split('.')[-1].strip()
        if new_format != org_format:
            audio = AudioSegment.from_file(audio_path)
            # export file as new format
            audio_path = audio_path.replace(org_format, new_format)
            audio.export(audio_path, format=new_format)
    except Exception as e:
        raise e
        
    try:
        sound = AudioSegment.from_file(audio_path, format='wav')
        sound = sound.set_frame_rate(freq)
        sound = sound.set_channels(1)
        
        if out_path is not None:
            audio_path = out_path
        sound.export(audio_path, format='wav')
    except Exception as e:
        raise e
        
    return audio_path



if __name__ == '__main__':
    from pathlib import Path
    from tqdm import tqdm
    
    audiofpths = [] 

    for path in tqdm(Path('dataset/RIRS_NOISES/').rglob('*.wav')):
        audiofpths.append(str(path))
        
    print(len(audiofpths))
    

    from multiprocessing import Pool


    # +
    p = Pool(processes=96)
    results = p.map(convert_audio, audiofpths)
    p.close()
    p.join()

    print('-----------')

