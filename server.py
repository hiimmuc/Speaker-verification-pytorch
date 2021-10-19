import io
import os
import time
from json import dumps
from pathlib import Path

from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
# from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

from config import ModelParams as pr
from model import SpeakerNet

app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'
UPLOAD_FOLDER = "static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['wav', 'mp4'])
api = Api(app)

# load model

model_path = str(Path('exp/ResNetSE34V2/model/best_state.model'))
kwrags = {'nOut': 512, 'nClasses': 400,
          'lr': 0.001, 'weight_decay': 0,
          'test_interval': 10, 'lr_decay': 0.95}
model = SpeakerNet(pr.save_path,
                   pr.model,
                   pr.optimizer,
                   pr.callbacks,
                   pr.criterion,
                   pr.device,
                   pr.max_epoch, **kwrags)
model.loadParameters(model_path)


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
    if request.method == 'POST':
        if 'files[]' not in request.files:
            flash('No file part')
            return redirect(request.url)
        files = request.files.getlist('files[]')
        audio = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                audio.append(str(Path(audio_path)))
                file.save(audio_path)
            else:
                return redirect('/')
        # predict
        t0 = time.time()
        # make prediction here
        audio_1 = audio[0]
        audio_2 = audio[1]
        result = model.pair_test(audio_1, audio_2, 100, 10, '', scoring_mode='cosine', cohorts=None)
        matching = result > 0.5
        # return and upload score
        inference_time = time.time() - t0

        return render_template('demo.html',
                               matching=str(bool(matching)),
                               result=str(round(result, 4)),
                               inference_time=str(round(inference_time, 4)))


if __name__ == '__main__':
    app.run(debug=True)
