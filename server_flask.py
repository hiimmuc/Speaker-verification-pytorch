import io
import os
import subprocess
import time
import wave
from argparse import Namespace
from json import dumps
from pathlib import Path

from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename

from model import SpeakerNet
from utils import *

app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['wav', 'mp3', 'flac'])
api = Api(app)

# load model
threshold = 0.50
model_path = str(Path('exp/RawNet2v2/model/last_state.model'))
config_path = str(Path('config_deploy.yaml'))
args = read_config(config_path)

model = SpeakerNet(**vars(args))
model.loadParameters(model_path)
model.eval()

# default enrollment
enroll_def = None


def convert_audio(audio_path, format='wav', freq=8000):
    # convert data to one form 16000Hz, only works on Linux

    print(f"Converting process...")
    if format != audio_path[-4:]:
        print("convert to wav")
        subprocess.call(['ffmpeg', '-i', audio_path,
                         audio_path[:-4] + format])

    print(f"Convert to {freq}Hz")
    fpath = str(audio_path).replace('(', '\(')
    fpath = str(Path(fpath.replace(')', '\)')))
    outpath = str(Path(fpath[:-4] + '_conv' + fpath[-4:]))
    out = subprocess.call(
        f'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar {freq} %s >/dev/null 2>/dev/null'
        % (fpath, outpath),
        shell=True)
    if out != 0:
        raise ValueError('Conversion failed %s.' % fpath)
    subprocess.call('rm %s' % (fpath), shell=True)
    subprocess.call('mv %s %s' % (outpath, fpath), shell=True)
    print('Done!')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


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
        if any(f'file{i}' not in request.files for i in [1, 2]):
            flash('No file part')
            return redirect(request.url)

        enroll = request.files['file1']
        test = request.files['file2']

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

        # check and convert audio:
        for audio_path in audio:
            with wave.open(audio_path, "rb") as wave_file:
                frame_rate = wave_file.getframerate()
            valid_audio = (audio_path[-4:] != '.wav') and frame_rate != args.sample_rate
            if not valid_audio:
                convert_audio(audio_path, format='wav', freq=args.sample_rate)

        # predict
        t0 = time.time()
        # make prediction here
        audio_1 = audio[0]
        audio_2 = audio[1]
        result = model.pair_test(audio_1, audio_2, 100, 10, '', scoring_mode='cosine', cohorts=None)
        matching = result > threshold
        # return and upload score
        inference_time = time.time() - t0
        os.remove(audio_2)  # remove compath

        return render_template('demo.html',
                               filename1=audio_1,
                               filename2=audio_2,
                               matching=str(bool(matching)),
                               result=str(round(result, 4)),
                               inference_time=str(round(inference_time, 4)))


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', port=8111)
    app.run(debug=True)
