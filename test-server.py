import base64
import enum
import io
import json
import os
import time
from argparse import Namespace
from json import dumps
from pathlib import Path

import numpy as np
import torch
from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from model import SpeakerNet
from utils import *
from server_utils import *

# check log folder exists
log_audio_path = str(Path('log_service/audio_v3_rd0311'))
os.makedirs(log_audio_path, exist_ok=True)
log_results_path = str(Path('log_service/results_v3_rd0311'))
os.makedirs(log_results_path, exist_ok=True)
log_audio_path_id = os.path.join(log_audio_path, "unknown_number")
os.makedirs(log_audio_path_id, exist_ok=True)

# ==================================================load Model========================================
sr = 8000
eval_frames=100
num_eval=20
normalize=True

threshold = 0.5437501668930054
fixed_threshold = 0.5

model_path = str(Path('backup/Raw_ECAPA/model/mix_0307_1357_v3.model'))
config_path = str(Path('backup/Raw_ECAPA/config_deploy.yaml'))
print("\n>Loaded from:", model_path, "with threshold:", threshold)

# read config and load model
args = read_config(config_path)

t0 = time.time()
model = SpeakerNet(**vars(args))
model.loadParameters(model_path, show_error=False)
model.eval()
print("Model Loaded time: ", time.time() - t0)

# ================================================Flask API=============================================
# Set up env for flask
app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DEBUG'] = True

api = Api(app)

# for matching call
@app.route('/isMatched', methods=['POST'])
def check_matching():
    audio_data = None
    
    current_time = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())).replace('-', '').replace(' ', '_').replace(':', '')
    cprint(text=f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]", fg='k', bg='g')
    ####################
    # Get request
    t0 = time.time()
    json_data = request.get_json()
    
    # print("\n> JSON <", json_data)
    if 'refSpeech' in json_data:
        data_json = json.loads(json_data)
        call_id = data_json['callId']
        phone = data_json['phone']
        ref_audio_data = data_json["refSpeech"]
        com_audio_data = data_json["comSpeech"]
        print("Got audio signal in", time.time() - t0, 'sec', end=' || ')
        
        # create dir to save
        phone = "unknown_number" if (len(phone)==0) else phone
        log_audio_path_id = os.path.join(log_audio_path, phone)
        log_result_id =  os.path.join(log_results_path, phone)
        
        os.makedirs(log_audio_path_id, exist_ok = True)
        os.makedirs(log_result_id, exist_ok = True)
    else:
        raise "Error: no data provide"
        
    print("Phone number:", phone)
    print("Number of samples Ref:", len(ref_audio_data), "Com:", len(com_audio_data))
    
    # convertstring of base64 to np array
    dtype = np.float64
    ref_audio_data_np = [decode_audio(audio_data, sr, dtype) for audio_data in ref_audio_data]
    com_audio_data_np = [decode_audio(audio_data, sr, dtype) for audio_data in com_audio_data]
    
    # preprcess audio
#     target_db = -10
#     ref_audio_data_np = [preprocess_audio(audio_data_np, target_db) for audio_data_np in ref_audio_data_np]
#     com_audio_data_np = [preprocess_audio(audio_data_np, target_db) for audio_data_np in com_audio_data_np]   
    
    ####################
    # save log audio 
    print("Saving audio files...")
    print("Audio informations: ")
    
    print(f"> Ref files:")
    for i, audio_data_np in enumerate(ref_audio_data_np):
        # check whether ref audio is exists
        print(len(audio_data_np)/sr, 's', end=' ')
        if not any(f'ref_{i}' in fname and str(call_id) in fname for fname in os.listdir(log_audio_path_id)):
            save_path = os.path.join(log_audio_path_id, f'{call_id}_{current_time}_ref_{i}.wav')
            sf.write(save_path, audio_data_np, sr)
            print(f"> Speaker footprint {i + 1}th saved info:", phone, call_id)
    
    print(f"\n> Com files:")
    for i, audio_data_np in enumerate(com_audio_data_np):
        print(len(audio_data_np)/sr, 's', end=' ')
        save_path = os.path.join(log_audio_path_id, f'{call_id}_{current_time}_com_{i}.wav')
        sf.write(save_path, audio_data_np, sr)
        
    ####################
    #  get embeddings each
    t = time.time()
    nom_confidence_scores, confidence_scores = compute_score_by_pair(model, ref_audio_data_np, com_audio_data_np,  
                                                                     threshold=threshold, fixed_threshold=fixed_threshold, 
                                                                     eval_frames=eval_frames, num_eval=num_eval,
                                                                     normalize=normalize, sr=sr)
    print(f"\nCompare by pair result: \nTotal time: {time.time()-t}s")
    print(*nom_confidence_scores, sep='\n')
    print(*confidence_scores, sep='\n')
    ######################
    # embed all and compare
#     t = time.time()
#     norm_mean_scores, mean_scores = compute_score_by_mean_ref(model, ref_audio_data_np, com_audio_data_np, 
#                                                               threshold=threshold, fixed_threshold=fixed_threshold, 
#                                                               eval_frames=eval_frames, num_eval=num_eval,
#                                                               normalize=normalize, sr=sr)
#     print(f"\nCompare by taking mean ref: \nTotal time: {time.time()-t}s")
#     print(norm_mean_scores)
#     print(mean_scores)
    ########################
    max_norm_score = np.mean(nom_confidence_scores)
#     max_norm_mean_score = np.amax(norm_mean_scores)
#     print(max_norm_score, max_norm_mean_score)
    final_score = max(max_norm_score, 0)
    
    # write log results
    with open(os.path.join(log_result_id, f"{call_id}.txt"), 'a') as wf:
        text = f">{current_time}<\nRef: \n" + ','.join([f'{current_time}_ref_{i}.wav' for i in range(len(ref_audio_data))]) + '\n'
        text += ("Com: \n" + ', '.join([f'{current_time}_com_{i}.wav' for i in range(len(com_audio_data))]) + '\n')
        text += (str(bool(final_score >= fixed_threshold)) + f"with score: {final_score}" + '\n>-------------------------------------------------<\n')
        wf.write(text)
        
    return jsonify({"isMatch": str(bool(final_score >= fixed_threshold)), "confidence": str(final_score), "Threshold": threshold})


@app.route('/embedding', methods=['POST'])
def get_embeding():
    audio_data = None    
    
    current_time = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())).replace('-', '').replace(' ', '_').replace(':', '')
    cprint(text=f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]", fg='k', bg='g')

    t0 = time.time()
    json_data = request.get_json()    
    # print("\n> JSON <", json_data)
    
    if 'base64Speech' in json_data:
        data_json = json.loads(json_data)
        audio_data = data_json["base64Speech"]
        call_id = data_json['callId']
        phone = data_json['phone']
        
        print("Got audio signal in", time.time() - t0, 'sec', end=' || ')
        #  create dir to save
        log_audio_path_id = os.path.join(log_audio_path, phone)
        os.makedirs(log_audio_path_id)
    else:
        raise "Error: no data provide"
    # convertstring of base64 to np array
    audio_data_np = decode_audio(audio_data, sr)
    
    t0 = time.time()
    save_path = os.path.join(log_audio_path_id, f'{call_id}_{current_time}_com_{i}.wav')

    if os.path.isfile(save_path):
        save_path = save_path.replace('ref', 'com')

    sf.write(save_path, audio_data_np, sr)
    print("Save audio signal to file:", save_path, round(time.time() - t0, 2), 's')

    t0 = time.time()
    emb = np.asarray(model.embed_utterance(audio_data_np, eval_frames=eval_frames, num_eval=num_eval, normalize=normalize, sr=sr))
    emb_json = json.dumps(emb.tolist())
    print("Inference time:", f"{time.time() - t0} sec", "|| Embeding size:", emb.shape)

    return jsonify({"Embedding": emb_json, "Inference_time": time.time() - t0, "Threshold": threshold})


@app.route('/', methods=['GET'])
def get_something():
    pass


if __name__ == '__main__':
    #     app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8111)