import base64
import io
import json
import os
import time
from argparse import Namespace
from json import dumps
from pathlib import Path

import numpy as np
from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from pydub import AudioSegment
from werkzeug.utils import secure_filename

from model import SpeakerNet
from utils import *

# ==================================================load Model========================================
# load model
threshold = 0.32077792286872864
model_path = str(Path('backup/Raw_ECAPA/model/best_state-235e-2.model'))
config_path = str(Path('backup/Raw_ECAPA/config_deploy.yaml'))
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


@app.route('/', methods=['POST'])
def get_embbeding():
    audio_data = None
    t0 = time.time()
    json_data = request.get_json()

    if 'data' in json_data:
        data_json = json.loads(json_data)
        audio_data = data_json["data"]
        sr = int(data_json["sample_rate"])
        print("Got audio signal", time.time() - t0)
    else:
        raise "Error: no data provide"
    # convertstring of base64 to np array
    audio_data_bytes = audio_data.encode('utf-8')
    audio_data_b64 = base64.decodebytes(audio_data_bytes)
    audio_data_np = np.frombuffer(audio_data_b64, dtype=np.float64)

#     # convert to AudioSegment
#     # ....
#     # none above, because we have convert the posted data to numpy array
# 
#     t0 = time.time()
#     sf.write("dataset/dump/dump.wav", audio_data_np,sr)
#     audio_path = "dataset/dump/dump.wav"
#     print("Save audio signal to dump files", audio_path, time.time() - t0)
#     audio_properties = get_audio_information(audio_path)
#     format = audio_path.split('.')[-1]
#     valid_audio = (audio_properties['sample_rate'] == args.sample_rate) and (format == args.target_format)
#     if not valid_audio:
#         audio_path = convert_audio(audio_path, new_format=args.target_format, freq=args.sample_rate)
    
    t0 = time.time()
    emb = np.asarray(model.embed_utterance(audio_data_np, eval_frames=100, num_eval=10, normalize=True, sr=sr))
    emb_json = json.dumps(emb.tolist())
    print("Inference time", time.time() - t0, "Embeding size", emb.shape)
    
    return jsonify({"Embedding": emb_json, "Inference_time": time.time() - t0, "Threshold": threshold})


@app.route('/', methods=['GET'])
def get_something():
    pass


if __name__ == '__main__':
#     app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8111)
