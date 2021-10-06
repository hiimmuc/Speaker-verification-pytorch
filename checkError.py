import time
from pathlib import Path
import glob
import os
import numpy as np
from tqdm.auto import tqdm
from utils import *
import itertools
import csv

def generate_checklist(raw_path):
        """
        Generate train test lists for zalo data
        """
        root = Path(raw_path)
        classpaths = [d for d in root.iterdir() if d.is_dir()]
        checklist = []
        checkdict = {}
        for classpath in classpaths:
            filepaths = list(classpath.glob('*.wav'))

            non_augment_path = list(
                filter(lambda x: 'augment' not in str(x), filepaths))
            
            label = str(non_augment_path[0].parent.stem.split('-')[0])

            checklist = [str(x).replace(raw_path, '') for x in non_augment_path[:]]
            checkdict[label] = list(itertools.combinations(checklist, 2))

        return checkdict

def convert_to_csv(checkdict, save_root):
    write_file = Path(save_root, 'checklist.csv')
    with open(write_file, 'w', newline='') as wf:
        spamwriter = csv.writer(wf, delimiter=',')
        spamwriter.writerow(['ref', 'com'])
        for k, v in checkdict.items():
            for v_ in v:
                spamwriter.writerow([v_[0], v_[1]])
    pass

def run_evaluate(check_list_csv):
    pass

def plot_grap(check_list_csv_result):
    pass 

dic = generate_checklist("dataset/wavs/")
# dic
for k, v in dic.items():
    print(k, ':', len(v))

convert_to_csv(dic, "dataset")

import matplotlib.pyplot as plot
from scipy.io import wavfile

def plot_spec(filepath):
    samplingFrequency, signalData = wavfile.read(filepath)



    # Plot the signal read from wav file

    plot.subplot(211)

    plot.title('Spectrogram of a wav file')



    plot.plot(signalData)

    plot.xlabel('Sample')

    plot.ylabel('Amplitude')



    plot.subplot(212)

    plot.specgram(signalData,Fs=samplingFrequency)

    plot.xlabel('Time')

    plot.ylabel('Frequency')



    plot.show()

import pandas as pd
df = pd.read_csv("dataset/checklist_result.csv")
c = 0
wrong_label = {}
for i, label in enumerate(list(df['label'])):
    if int(label) == 0:
        c += 1
#         print(df['audio_1'][i], '||', df['audio_2'][i], ':', df['score'][i])
        
        if df['audio_1'][i] not in wrong_label:
            wrong_label[df['audio_1'][i]] = 0
        if df['audio_2'][i] not in wrong_label:
            wrong_label[df['audio_2'][i]] = 0
        wrong_label[df['audio_1'][i]] += 1
        wrong_label[df['audio_2'][i]] += 1
        
print("Total wrong label:", c)
print("===================================================================")
c = 0
for i, score in enumerate(list(df['score'])):
    if  score < 0.4:
        c += 1
#         print(df['audio_1'][i], '||', df['audio_2'][i], df['score'][i])
print('Total wrong score:', c)
print('================================================================')
# sort wrong label

wrong_label = {k: v for k, v in sorted(wrong_label.items(), key=lambda item: item[1], reverse=True)}
for k, v in wrong_label.items():
    if v > 1:
        print(k, ':', v)

glob.glob('dataset/wavs/561-M-45/*.wav')

import IPython.display as ipd
path = "dataset/wavs/561-M-45/561-M-45-15.wav"
plot_spec(path)
ipd.Audio(path)

path = "dataset/wavs/561-M-45/561-11.wav" # noise
plot_spec(path)
ipd.Audio(path)
