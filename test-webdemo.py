import io
import os
import time

from argparse import Namespace
from json import dumps
from pathlib import Path

from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

from model import SpeakerNet
from utils import *



# ==================================================Begin========================================
# Set up env for flask 
app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'

UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'flac'])

api = Api(app)

# load model
threshold = 0.27179449796676636
model_path = str(Path('exp/dump/RawNet2v3/model/best_state_5000spk.model'))
config_path = str(Path('config_deploy.yaml'))
args = read_config(config_path)

t0 = time.time()
model = SpeakerNet(**vars(args))
model.loadParameters(model_path, show_error=False)
model.eval()
print("Model Loaded time: ", time.time() - t0)

# default enrollment
enroll_def = None


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
           
# =============================================Flask API==================================
@app.route('/')
def home():
    return render_template('demo.html')


@app.route('/audio/<filename>')
def audio(filename):
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(audio_path)


@app.route('/', methods=['GET', 'POST'])
def upload_audio():
    global enroll_def
    
    if request.method == 'POST':
        t0 = time.time()
        if any(f'file{i}' not in request.files for i in [1, 2]):
            flash('No file part')
            return redirect(request.url)

        enroll = request.files['file1']
        test = request.files['file2']
        
        # to make the fisrt enrollment be the base user and the other be the test user, if change user, remove the last one
        if allowed_file(enroll.filename) and enroll:
            enroll_def = enroll
        else:
            enroll = enroll_def


        files = [enroll, test]

        audio = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio.append(str(Path(audio_path)))
                if not os.path.exists(audio_path):
                    file.save(audio_path)
            else:
                return redirect('/')
        print("Upload files times:", time.time() - t0)
        
        # check and convert audio:
        for i, audio_path in enumerate(audio):
            t0 = time.time()
            audio_properties = get_audio_information(audio_path)
            format = audio_path.split('.')[-1]
            valid_audio = (audio_properties['sample_rate'] == args.sample_rate) and (format == args.target_format)
            if not valid_audio:
                audio[i] = audio_path = convert_audio(audio_path, new_format=args.target_format, freq=args.sample_rate)
            print("Check validation time:", time.time() - t0)
                
        # predict
        t0 = time.time()
        # make prediction here
        audio_1 = audio[0]
        audio_2 = audio[1]
        
        result = model.pair_test(audio_1, audio_2, 100, 10, '', scoring_mode='cosine', cohorts=None)

        ratio = threshold / 0.5
        result = (result / ratio) if (result / ratio) < 1 else 1
        matching = result > 0.5
        
        # return and upload score
        inference_time = time.time() - t0
        print("Predict time", inference_time)
        
        if os.path.exists(audio_2):
            os.remove(audio_2) # remove the test voice audio to prevent overloading

        return render_template('demo.html',
                               filename1=audio_1,
                               filename2=audio_2,
                               matching=str(bool(matching)),
                               result=str(round(result, 4)),
                               inference_time=str(round(inference_time, 4)))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8111)
