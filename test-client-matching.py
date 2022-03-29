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

from utils import cprint
from pydub import AudioSegment
from processing.wav_conversion import normalize_audio_amp

ref1 = "log_service/audio_bu/ref_bu/auth_ref/366524143/366524143-2022-03-15-08-34-14.wav"
ref2 = "log_service/audio_bu/ref_bu/auth_ref/366524143/366524143-2022-03-15-08-34-25.wav"
ref3 = "log_service/audio_bu/ref_bu/auth_ref/366524143/366524143-2022-03-15-08-34-39.wav"

ref = ['log_service/audio_thudth1/audio_thudth1/ref_audio/988988883/988988883-2022-03-24-10-31-23.wav',
       'log_service/audio_thudth1/audio_thudth1/ref_audio/988988883/988988883-2022-03-24-10-31-36.wav',
       'log_service/audio_thudth1/audio_thudth1/ref_audio/988988883/988988883-2022-03-24-10-31-50.wav']

ref_norm = ['log_service/audio_thudth1/audio_thudth1/ref_audio_norm/988988883/988988883-2022-03-24-10-31-23.wav',     
            'log_service/audio_thudth1/audio_thudth1/ref_audio_norm/988988883/988988883-2022-03-24-10-31-36.wav',
            'log_service/audio_thudth1/audio_thudth1/ref_audio_norm/988988883/988988883-2022-03-24-10-31-50.wav']

com = ['log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-32-44.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-00.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-19.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-35.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-55.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-34-12.wav',]

com_norm = ['log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-32-44_norm.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-00_norm.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-19_norm.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-35_norm.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-33-55_norm.wav',
 'log_service/audio_thudth1/audio_thudth1/logs/988988883/988988883-988988883-20220324-173219-2022-03-24-10-34-12_norm.wav']


URL = "http://0.0.0.0:8111/isMatched"  # http://10.254.136.107:8111/

# from processing.wav_conversion
#
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
    try:
        r = requests.post(URL, json=data_json)
        # print with color state of response
        print("Connection: ", end='')
        state = "Success" if int(r.status_code) == 200 else "Failed"
        color_text = 'g' if int(r.status_code) == 200 else 'r'
        cprint(text=state, fg=color_text, end=' ')

        response = r.json()
        print("Response time:", time.time() - t)

        isMatch = response["isMatch"]
        print("isMatch:", end=' ')

        color_text = 'g' if (isMatch) == 'True' else 'r'
        cprint(text=str(isMatch), fg=color_text, end=' ') # print with color match state

        confidence = response["confidence"]
        print("Confidence:", confidence)
        
    except Exception as e:
            print("Error when getting response ::: " + str(e))
        


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
        refs = [args.ref]
        
    if args.com:
        coms = [args.com]
        
    # test log
    for refs in [ref, ref_norm]:
        for coms in [com, com_norm]:
            print(refs, end=' vs ')
            for c in coms:
                print(c)
                get_response(refs, [c])
            
    # get_response(refs, coms)
    print('')
#     get_response(refs, coms)
######################################################################