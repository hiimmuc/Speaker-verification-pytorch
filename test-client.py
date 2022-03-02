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
URL = "http://192.168.0.4:8111/Embedding/"  # http://10.254.136.107:8111/


def check_matching(ref_emb, com_emb, threshold=0.5):
    score = cosine_simialrity(ref_emb, com_emb)
    ratio = threshold / 0.5
    result = (score / ratio) if (score / ratio) < 1 else 1
    matching = result > 0.5
    print("Result:", result, "Score:", score)
    return matching


def encode_audio(path):
    audio, sr = sf.read(str(Path(path)))
    audio = audio.astype(np.float64)
    audio_signal = base64.b64encode(audio)
    return audio_signal, sr


def get_response(path):
    t = time.time()
    signal, sr = encode_audio(path)
    data = {'callId': '366524143-20211229-100000',
            'phone': '366524143',
            'base64Speech': signal,
            'sample_rate': sr}
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

    if 'Embedding' in response:
        embeding_vec_str = response["Embedding"]
        embeding_vec_np = np.asanyarray(json.loads(embeding_vec_str), dtype=np.float64)
        embeding_vec_tensor = torch.from_numpy(embeding_vec_np)
    else:
        embeding_vec_tensor = torch.from_numpy(np.zeros((10, 512)))

    if 'Threshold' in response:
        threshold = float(response["Threshold"])
    else:
        threshold = 0.5

    return (embeding_vec_tensor, threshold)


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
    emb_ref, threshold = get_response(args.ref)
    emb_com, threshold = get_response(args.com)

#     print(type(emb_ref), emb_ref.shape)
#     print(type(emb_com), emb_com.shape)
    cprint(text="\n> RESULTS <", fg='k')

    matched = check_matching(emb_ref, emb_com, threshold)

    print("Matching:", end=' ')

    if matched:
        cprint(text=str(matched), fg='g')
    else:
        cprint(text=str(matched), fg='r')

    print("Total time:", time.time() - t, 'sec\n=========================================\n')
