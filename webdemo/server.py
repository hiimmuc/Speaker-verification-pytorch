import io
import os
import time
from json import dumps

from flask import (Flask, Markup, flash, jsonify, redirect, render_template,
                   request, send_file, send_from_directory, url_for)
from flask_restful import Api, Resource
# from model import SpeakerNet
from sqlalchemy import create_engine
from werkzeug.utils import secure_filename

from sqlconnect import *

app = Flask(__name__, template_folder='templates')
app.secret_key = 'super secret key'
UPLOAD_FOLDER = "webdemo/static/uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['wav', 'mp4'])
api = Api(app)
# load model
# model = SpeakerNet()


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
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            else:
                return redirect('/')
        # predict
        # make prediction here
        # return and upload score
        t0 = time.time()
        result = None
        inference_time = time.time() - t0

        return render_template('demo.html',
                               result=result,
                               inference_time=inference_time)


if __name__ == '__main__':
    app.run(debug=True)
