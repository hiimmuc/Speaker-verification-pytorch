# client
import json

import requests
import simplejson

import soundfile as sf
import numpy as np
import base64


def encode_audio(path):
    audio, sr = sf.read(path)
    audio = audio.astype(np.float64)
    audio_signal = base64.b64encode(audio)
    return audio_signal, sr

if __name__ == '__main__':
    signal, sr = encode_audio("dataset/dump/speaker_272-10_augmented_1.wav")
    data = {'data': signal, 'sample_rate': sr}
    data_json = simplejson.dumps(data)
    
    r = requests.post("http://0.0.0.0:8111/", json=data_json)
    print(r.status_code)
    print(r.json())
