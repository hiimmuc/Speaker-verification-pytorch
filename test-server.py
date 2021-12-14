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

# from model import SpeakerNet
from utils import *

# ==================================================load Model========================================
# load model
# threshold = 0.27179449796676636
# model_path = str(Path('exp/dump/RawNet2v3/model/best_state_5000spk.model'))
# config_path = str(Path('config_deploy.yaml'))
# args = read_config(config_path)

# t0 = time.time()
# model = SpeakerNet(**vars(args))
# model.loadParameters(model_path, show_error=False)
# model.eval()
# print("Model Loaded time: ", time.time() - t0)

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
    print(json_data, type(json_data))
    if 'data' in json_data:
        audio_data = json.loads(json_data)["data"]
    else:
        raise "Error: no data provide"
    audio_data_bytes = audio_data.encode('utf-8')
    audio_data_b64 = base64.decodebytes(audio_data_bytes)
    audio_data_np = np.frombuffer(audio_data_b64, dtype=np.float64)
    print(audio_data_np, type(audio_data_np))
    audio_data_np_json = json.dumps(audio_data_np.tolist())


#     # convert to AudioSegment
#     # ....
#     # enter above

#     assert isinstance(audio_data, AudioSegment)
#     filename = f"time.strftime("%Y-%m-%d %H:%M:%S").wav"
#     audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

#     audio_data.export(audio_path, format='wav')
#     audio_properties = get_audio_information(audio_path)
#     format = audio_path.split('.')[-1]
#     valid_audio = (audio_properties['sample_rate'] == args.sample_rate) and (format == args.target_format)
#     if not valid_audio:
#         audio_path = convert_audio(audio_path, new_format=args.target_format, freq=args.sample_rate)

#     return np.asarray(model.embed_utterance(audio_path, eval_frames=100, num_eval=10, normalize=True))
    return jsonify({"data": audio_data_np_json, "time": time.time() - t0})


@app.route('/', methods=['GET'])
def get_something():
    pass


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0', port=8111)
