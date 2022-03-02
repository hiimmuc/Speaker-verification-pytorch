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
from utils.utils import *

# ==================================================load Model========================================
# load model
# backup/Raw_ECAPA/model/best_state-CB_final_v1.model # domain callbot 0.38405078649520874
# backup/Raw_ECAPA/model/best_state-train_mix_2domains.model # mix 0.3097502887248993
# backup/Raw_ECAPA/model/best_state-235e-2.model # domain cskh 0.19131483137607574 0.21517927944660187

threshold = 0.21517927944660187
model_path = str(Path('backup/Raw_ECAPA/model/best_state-278e-2.model'))
config_path = str(Path('backup/Raw_ECAPA/config_deploy.yaml'))
log_audio_path = str(Path('log_service/audio'))
os.makedirs(log_audio_path, exist_ok=True)
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

    current_time = str(time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())).replace('-', '').replace(' ', '_').replace(':', '')
    cprint(text=f"\n[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}]", fg='k', bg='g')

    t0 = time.time()

    json_data = request.get_json()

    print("\n> JSON <", json_data)
    if 'base64Speech' in json_data:
        data_json = json.loads(json_data)
        audio_data = data_json["base64Speech"]
        sr = int(data_json["sample_rate"])
        print("Got audio signal in", time.time() - t0, 'sec', end=' || ')
    else:
        raise "Error: no data provide"
    # convertstring of base64 to np array
    audio_data_bytes = audio_data.encode('utf-8')
    audio_data_b64 = base64.decodebytes(audio_data_bytes)
    audio_data_np = np.frombuffer(audio_data_b64, dtype=np.float64)
    print("Audio duration:", len(audio_data_np)/sr, 's')

#     # convert to AudioSegment
#     # ....
#     # none above, because we have convert the posted data to numpy array
#
    t0 = time.time()
    save_path = os.path.join(log_audio_path, f'{current_time}_ref.wav')
    if os.path.isfile(save_path):
        save_path = save_path.replace('ref', 'com')

    sf.write(save_path, audio_data_np, sr)
    print("Save audio signal to file:", save_path, time.time() - t0)

#     audio_properties = get_audio_information(audio_path)
#     format = audio_path.split('.')[-1]
#     valid_audio = (audio_properties['sample_rate'] == args.sample_rate) and (format == args.target_format)
#     if not valid_audio:
#         audio_path = convert_audio(audio_path, new_format=args.target_format, freq=args.sample_rate)
#     no need to check validity of audio

    t0 = time.time()
    emb = np.asarray(model.embed_utterance(audio_data_np, eval_frames=100, num_eval=20, normalize=True, sr=sr))
    emb_json = json.dumps(emb.tolist())
    print("Inference time:", f"{time.time() - t0} sec", "|| Embeding size:", emb.shape)

    return jsonify({"Embedding": emb_json, "Inference_time": time.time() - t0, "Threshold": threshold})


@app.route('/', methods=['GET'])
def get_something():
    pass


if __name__ == '__main__':
    #     app.run(debug=True)
    app.run(debug=True, host='0.0.0.0', port=8111)
