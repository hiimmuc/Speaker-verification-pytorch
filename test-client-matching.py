# client
import argparse
import base64
import json
import time
from pathlib import Path

import numpy as np
import requests
import simplejson
import soundfile as sf
import torch

from utils.utils import *

default_path1 = "dataset/train_callbot/325002523/325002523-20211229-143811-in_0.wav"
default_path2 = "dataset/train_callbot/325002523/325002523-20220107-163500-in_0.wav"
URL = "http://0.0.0.0:8111/isMatched/"  # http://10.254.136.107:8111/


def encode_audio(path):
    audio, sr = sf.read(str(Path(path)))
    audio = audio.astype(np.float64)
    audio_signal = base64.b64encode(audio)
    return audio_signal, sr


def get_response(path1, path2):
    t = time.time()
    signal1, sr1 = encode_audio(path1)
    signal2, sr2 = encode_audio(path2)
    assert sr1 == sr2, "Sample rate not equal"

    data = {'callId': '366524143-20211229-100000',
            'phone': '366524143',
            'refSpeech': [signal1,
                          signal1,
                          signal1],
            'comSpeech': [signal2,
                          signal2,
                          signal2],
            'sample_rate': sr1}

    data_json = simplejson.dumps(data)

    r = requests.post(URL, json=data_json)
    print("Success: ", end='')
    color_text = 'g' if int(r.status_code) == 200 else 'r'
    cprint(text=str(int(r.status_code) == 200), fg=color_text)

    response = r.json()
    print("Response time:", time.time() - t)

    if 'Inference_time' in response:
        infer_time = float(response["Inference_time"])
        print("Predict time:", infer_time)
    else:
        infer_time = 0.0

    if "isMatch" in response:
        isMatch = response["isMatch"]
        print("isMatch:", isMatch)
    if "confidence" in response:
        confidence = response["confidence"]
        print("confidence:", confidence)

    # if 'Embedding' in response:
    #     embeding_vec_str = response["Embedding"]
    #     embeding_vec_np = np.asanyarray(json.loads(embeding_vec_str), dtype=np.float64)
    #     embeding_vec_tensor = torch.from_numpy(embeding_vec_np)
    # else:
    #     embeding_vec_tensor = torch.from_numpy(np.zeros((10, 512)))

    # if 'Threshold' in response:
    #     threshold = float(response["Threshold"])
    # else:
    #     threshold = 0.5


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TestService")
    parser.add_argument('--ref', '-r',
                        type=str,
                        default=default_path1,
                        help='path to file 1')
    parser.add_argument('--com', '-c',
                        type=str,
                        default=default_path2,
                        help='path to file 2')
    args = parser.parse_args()

    t = time.time()
    print(f"<[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]>")

    print("Getting response...")
    get_response(args.ref, args.com)
