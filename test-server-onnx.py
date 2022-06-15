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
import soundfile as sf

# from model import SpeakerEncoder, WrappedModel, ModelHandling
import onnxruntime
from utils import (read_config, cprint)
from server_utils import *

# check log folder exists
log_service_root = str(Path('log_service/'))
os.makedirs(log_service_root, exist_ok=True)

# ==================================================load Model========================================
norm_mode = 'uniform'
base_threshold = 0.5
compare_threshold = 0.6

threshold = 0.30186375975608826
model_path = str(Path('backup/1001/Raw_ECAPA/ARmSoftmax/model/best_state_top4.pt'))
config_path = str(Path('yaml/configuration.yaml'))
args = read_config(config_path)

print("\n<<>> Loaded from:", model_path, "with threshold:", threshold)

# read config and load model
args = read_config(config_path)
args = Namespace(**args)

sr = args.audio_spec['sample_rate']
num_eval = args.num_eval
normalize=True
##
t0 = time.time()
net = WrappedModel(SpeakerEncoder(**vars(args)))
max_iter_size = args.step_size
speaker_model = ModelHandling(
        net, **dict(vars(args), T_max=max_iter_size))
speaker_model.loadParameters(model_path, show_error=False)
speaker_model.__model__.eval()


# def to_numpy(tensor):
#     if not torch.is_tensor(tensor):
#         tensor = torch.FloatTensor(tensor)
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# onnx_session = onnxrt.InferenceSession(model_path)
# onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(inp)}
# onnx_output = onnx_session.run(None, onnx_inputs)

print("Model Loaded time: ", time.time() - t0)

# ================================================Flask API=============================================
# Set up env for flask
app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'

app.config['UPLOAD_FOLDER'] = log_service_root
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['DEBUG'] = True

api = Api(app)

# for matching call
@app.route('/isMatched', methods=['POST'])
def check_matching():
    audio_data = None
    
    current_day = str(time.strftime('%Y-%m-%d', time.gmtime())).replace('-', '').replace(' ', '_').replace(':', '')
    # create log dir
    log_audio_path = str(Path(f'log_service/{current_day}/audio'))
    os.makedirs(log_audio_path, exist_ok=True)
    log_results_path = str(Path(f'log_service/{current_day}/results'))
    os.makedirs(log_results_path, exist_ok=True)
    log_audio_path_id = os.path.join(log_audio_path, "unknown_number")
    os.makedirs(log_audio_path_id, exist_ok=True)
    #
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
        
    print("Phone number:", phone, end=' || ')
    print("Number of samples: Ref", len(ref_audio_data), "Com", len(com_audio_data))
    
    # convertstring of base64 to np array
    dtype = np.float64
    ref_audio_data_np = [decode_audio(audio_data, args.audio_spec, dtype) for audio_data in ref_audio_data]
    com_audio_data_np = [decode_audio(audio_data, args.audio_spec, dtype) for audio_data in com_audio_data]
    
    ## preprcess audio
    # target_db = -10
    # ref_audio_data_np = [preprocess_audio(audio_data_np, target_db) for audio_data_np in ref_audio_data_np]
    # com_audio_data_np = [preprocess_audio(audio_data_np, target_db) for audio_data_np in com_audio_data_np]   
    
    ####################
    # save log audio 
    print("Saving audio files...")   
    print(f"> Ref files:")
    log_audio_path_id_ref = os.path.join(log_audio_path_id, 'ref')
    os.makedirs(log_audio_path_id_ref, exist_ok=True)
    
    if not any(str(call_id) in fname for fname in os.listdir(log_audio_path_id_ref)):
        for i, audio_data_np in enumerate(ref_audio_data_np):
            # check whether ref audio is exists
            print(len(audio_data_np)/sr, 's', end=' ')
            save_path = os.path.join(log_audio_path_id_ref, f'{call_id}_ref_{i}.wav')
            sf.write(save_path, audio_data_np, sr)
            print(f"> Speaker footprint {i + 1}th saved info:", phone, call_id)
    
    print(f"> Com files:", end=' ')
    log_audio_path_id_com = os.path.join(log_audio_path_id, 'com')
    os.makedirs(log_audio_path_id_com, exist_ok=True)
    for i, audio_data_np in enumerate(com_audio_data_np):
        print(len(audio_data_np)/sr, 's', end=' ')
        save_path = os.path.join(log_audio_path_id_com, f'{call_id}_com_{i}.wav')
        sf.write(save_path, audio_data_np, sr)
        
    ####################
    #  get embeddings each
    t = time.time()
    print("\n\nInference results:")
    print(f"Compare by pair: ", end='')
    nom_confidence_scores, confidence_scores = compute_score_by_pair(speaker_model, ref_audio_data_np, com_audio_data_np,  
                                                                     threshold=threshold, base_threshold=base_threshold, 
                                                                     num_eval=num_eval,
                                                                     normalize=normalize, 
                                                                     norm_mode=norm_mode)
    print(f"\\Score: {confidence_scores} -> {nom_confidence_scores} \\Total time: {round(time.time()-t, 4)}s")
    
    ######################
    # embed all and compare
    t = time.time()
    print(f"Compare by mean ref: ", end='')
    norm_mean_emb_scores, mean_emb_scores = compute_score_by_mean_ref(speaker_model, ref_audio_data_np, com_audio_data_np, 
                                                                      threshold=threshold, base_threshold=base_threshold, 
                                                                      num_eval=num_eval,
                                                                      normalize=normalize, 
                                                                      norm_mode=norm_mode)
    
    print(f"\\Score: {mean_emb_scores} -> {norm_mean_emb_scores} \\Total time: {round(time.time()-t, 4)}s")
    
    ######################## Processing scores
    score_lst = np.concatenate((confidence_scores, mean_emb_scores), axis=None)
    norm_score_lst = np.concatenate((nom_confidence_scores, norm_mean_emb_scores), axis=None)
    
    mean_confidence_scores = np.mean(confidence_scores)
    mean_nom_confidence_scores = np.mean(nom_confidence_scores)
    
    mean_mean_emb_scores = np.mean(mean_emb_scores)
    mean_norm_mean_emb_scores = np.mean(norm_mean_emb_scores)
    
    max_norm_mean_score = np.amax(norm_mean_emb_scores)
    mean_overall_score = np.mean([mean_confidence_scores, mean_mean_emb_scores])
    #
    # if mean of confidences scores of 3 pairs tests is greater than the based threshold, suppose to be verified
    final_score = np.amax(norm_score_lst) if (float(mean_confidence_scores) > threshold) else np.mean(norm_score_lst)
    # final_score = np.mean(norm_score_lst)
    ########################
    print('Verifed: ', end='')
    color = 'r' if not bool(final_score >= compare_threshold) else 'g'
    cprint(text=str(bool(final_score >= compare_threshold)), fg=color)
    
    # write log results
    with open(os.path.join(log_result_id, f"{call_id}.txt"), 'a') as wf:
        text = f">{current_day}<\nRef: \n" + ','.join([f'{call_id}_ref_{i}.wav' for i in range(len(ref_audio_data))]) + '\n'
        text += ("Com: \n" + ', '.join([f'{call_id}_com_{i}.wav' for i in range(len(com_audio_data))]) + '\n')
        text += (str(bool(final_score >= compare_threshold)) + f"with score: {final_score}" + '\n>-------------------------------------------------<\n')
        wf.write(text)
        
    return jsonify({"isMatch": str(bool(final_score >= compare_threshold)), 
                    "confidence": str(final_score), 
                    "Threshold": threshold})


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
        log_audio_path_id_emb = os.path.join(log_audio_path, phone, 'emb')
        os.makedirs(log_audio_path_id_emb, exist_ok=True)
    else:
        raise "Error: no data provide"
    # convertstring of base64 to np array
    audio_data_np = decode_audio(audio_data, sr)
    
    t0 = time.time()
    save_path = os.path.join(log_audio_path_id_emb, f'{call_id}_{current_time}_ref.wav')

    if os.path.isfile(save_path):
        save_path = save_path.replace('ref', 'com')

    sf.write(save_path, audio_data_np, sr)
    print("Save audio signal to file:", save_path, round(time.time() - t0, 2), 's')

    t0 = time.time()
    emb = np.asarray(speaker_model.embed_utterance(audio_data_np, eval_frames=eval_frames, num_eval=num_eval, normalize=normalize, sr=sr))
    emb_json = json.dumps(emb.tolist())
    print("Inference time:", f"{time.time() - t0} sec", "|| Embeding size:", emb.shape)

    return jsonify({"Embedding": emb_json, "Inference_time": time.time() - t0, "Threshold": threshold})


@app.route('/', methods=['GET'])
def get_something():
    """dump script"""
    pass


if __name__ == '__main__':
    #     app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8111)