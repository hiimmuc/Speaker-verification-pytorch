# client
import json

import requests
import simplejson

import soundfile as sf
import numpy as np
import base64
from utils import *
import argparse
import time
import torch

parser = argparse.ArgumentParser(description="TestService")
default_path1 = "dataset/dump/thuyth.wav"
default_path2 = "dataset/dump/366524143-20211229-084534_5.wav"
URL = "http://0.0.0.0:8111/"

def check_matching(ref_emb, com_emb, threshold=0.5):
    score = cosine_simialrity(ref_emb, com_emb)
    ratio = threshold / 0.5
    result = (score / ratio) if (score / ratio) < 1 else 1
    matching = result > 0.5
    print("Result:", result, "Score:", score)
    return matching

def encode_audio(path):
    audio, sr = sf.read(path)
    audio = audio.astype(np.float64)
    audio_signal = base64.b64encode(audio)
    return audio_signal, sr

def get_response(path):
    t = time.time()
    signal, sr = encode_audio(path)
    data = {'data': signal, 'sample_rate': sr}
    data_json = simplejson.dumps(data)
    
    r = requests.post(URL, json=data_json)
    print(r.status_code)
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
    
    parser.add_argument('--ref', '-r',
                        type=str,
                        default=default_path1,
                        help='path to file 1')
    parser.add_argument('--com', '-c',
                        type=str,
                        default=default_path2,
                        help='path to file 2')
    args = parser.parse_args()
    
    emb_ref, threshold = get_response(args.ref)
    emb_com, threshold = get_response(args.com)
    
#     print(type(emb_ref), emb_ref.shape)
#     print(type(emb_com), emb_com.shape)
   
    print("Matching:", check_matching(emb_ref, emb_com, threshold))

        

    
