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

from utils import *

ref1 = "log_service/audio_v3_rd0311/912582757/912582757-20220313-215037_20220313_145104_ref_0.wav"
ref2 = "log_service/audio_v3_rd0311/912582757/912582757-20220313-215037_20220313_145104_ref_1.wav"
ref3 = "log_service/audio_v3_rd0311/912582757/912582757-20220313-215037_20220313_145104_ref_2.wav"

# com_true_0 = "log_service/audio_v2/912582757/912582757-20220311-094535_20220311_024610_com_0.wav" # 1 same id low vol
# com_true_1 = "log_service/audio_v2/912582757/912582757-20220311-094535_20220311_024610_com_0_rm_noise.wav" # 1 same id low vol

# com_false_0 = "log_service/audio_v2/912582757/912582757-20220307-170310_20220307_100403_com_0.wav" # 0 same id wrong spk
# com_false_1 = "log_service/audio_v2/912582757/912582757-20220307-170310_20220307_100403_com_0_rm_noise.wav" # 0 same id wrong spk
# com_false_2 = "log_service/audio_v2/346165056/346165056-20220307-172258_20220307_102351_com_0.wav" # 0 diff id

# com_true_2 = "log_service/audio_v2/912582757/912582757-20220309-091854_20220309_021947_com_0.wav" # 1 same id 
# com_true_3 = "log_service/audio_v2/912582757/912582757-20220310-172944_20220310_103009_com_0.wav" # 1 same id noise
# com_true_4 = "log_service/audio_v2/912582757/912582757-20220310-172944_20220310_103009_com_0_rm_noise.wav" # 1 same id noise
# com_true_5 = "log_service/audio_v2/912582757/912582757-20220311-094535_20220311_024558_ref_2_rm_noise.wav"
# com_true_6 = "log_service/audio_v2/912582757/912582757-20220311-094535_20220311_024558_ref_2.wav"
com_true = ["log_service/audio_v3_rd0311/912582757/912582757-20220311-155624_20220311_085755_com_0.wav", 
            "log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090412_com_0.wav", 
            "log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091059_com_0.wav"]

com_false = ['log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_085929_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_085957_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_090009_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090520_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_090957_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091012_com_0.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091217_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091231_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091243_com_0.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091258_com_0.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091322_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090307_com_0.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090319_com_0.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090331_com_0.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-175108_20220311_105118_com_0.wav']

com_true_rn = ["log_service/audio_v3_rd0311/912582757/912582757-20220311-155624_20220311_085755_com_0_rm_noise.wav", 
            "log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090412_com_0_rm_noise.wav", 
            "log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091059_com_0_rm_noise.wav"]
com_false_rn = ['log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_085929_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_085957_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-155904_20220311_090009_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090520_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_090957_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091012_com_0_rm_noise.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091217_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091231_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091243_com_0_rm_noise.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091258_com_0_rm_noise.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160918_20220311_091322_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090307_com_0_rm_noise.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090319_com_0_rm_noise.wav',
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-160203_20220311_090331_com_0_rm_noise.wav' ,
             'log_service/audio_v3_rd0311/912582757/912582757-20220311-175108_20220311_105118_com_0_rm_noise.wav']


URL = "http://0.0.0.0:8111/isMatched"  # http://10.254.136.107:8111/

# include from utils
# def normalize_audio(signal):
#     try:
#         intinfo = np.iinfo(signal.dtype)
#         return signal / max( intinfo.max, -intinfo.min )

#     except ValueError: # array is not integer dtype
#         return signal / max( signal.max(), -signal.min())

def encode_audio(path):
    # audio, sr = sf.read(str(Path(path)))
    # segment -> np -> base64 -> b64 str
    sr = 8000
    audio_seg = AudioSegment.from_file(path)
    # convert to numpy
    audio = audio_seg.get_array_of_samples()
    audio = np.array(audio).astype(np.float64)
    audio = normalize_audio_amp(audio)
    # encode base64 str format
    audio_signal_bytes = base64.b64encode(audio)
    audio_signal_str = audio_signal_bytes.decode('utf-8')
    return audio_signal_str, sr

def get_response(refs, coms):
    t = time.time()
    signal_refs = [encode_audio(path)[0] for path in refs]
    signal_coms = [encode_audio(path)[0] for path in coms]
    
    data = {'callId': 'test_audio',
            'phone': '',
            'refSpeech': signal_refs,
            'comSpeech': signal_coms}

    data_json = json.dumps(data)

    r = requests.post(URL, json=data_json)
    # print with color state of response
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
        print("isMatch:", end=' ')
        # print with color match state
        color_text = 'g' if (isMatch) == 'True' else 'r'
        cprint(text=str(isMatch), fg=color_text)
    if "confidence" in response:
        confidence = response["confidence"]
        print("confidence:", confidence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="TestService")
    parser.add_argument('--ref', '-r',
                        type=str,
                        default=None,
                        help='path to file 1')
    parser.add_argument('--com', '-c',
                        type=str,
                        default=None,
                        help='path to file 2')
    args = parser.parse_args()

    t = time.time()
    
    print(f"<[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]>")
    print("Getting response...")
    if args.ref:  
        refs = [args.ref] * 3
    else:
        refs = [ref1, ref2, ref3]
    if args.com:
        coms = [args.com]
    else:
        for i, com in enumerate(com_true_rn):
            print(1 + i, True)
            get_response(refs, [com])
        for i, com in enumerate(com_false_rn):
            print(1 + i, False)
            get_response(refs, [com])
    print('')
#     get_response(refs, coms)
######################################################################