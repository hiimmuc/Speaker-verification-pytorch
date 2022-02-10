import os
import csv
import glob
import time
import numpy as np

import wave
import librosa
import contextlib
import random
import shutil
import subprocess
import soundfile as sf

from tqdm import tqdm
from pathlib import Path
from scipy.io import wavfile

from utils import *


def get_duration_file(fn_audio):
    with contextlib.closing(wave.open(str(fn_audio),'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
    return duration

def get_infor_file(fn_audio):
    ob =  sf.SoundFile(fn_audio)
    return ob.subtype, ob.samplerate, ob.channels


def get_amplitute_file(path):
    sr, data = wavfile.read(path)
#     bit_depth = int(get_infor_file(path)[0].split('_')[-1])
    bit_depth = 16
    db = 20 * np.log10(max(abs(data))/(2**(bit_depth - 1) - 1))
    return sr, min(data), max(data), db

def get_duration_folder(folder):
    total_length = 0
    for audio in glob.glob(f"{folder}/*.wav"):
        try:
            total_length += get_duration_file(audio)
        except:
            print("error in ",audio)
    return total_length


def get_size_file(fname):
    return Path(fname).stat().st_size

def get_size_folder(folder):
    return sum([float(get_size_file(f)) for f in glob.glob(f"{folder}/*")])


def get_audio_information_stats(filename):
    cmd = ['ffmpeg', '-i', filename, '-map', '0:a', '-af', 'astats', '-f', 'null', '-']
#     cmd = ['ffprobe', filename]
    out = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stderr
    output_lines = [line.strip() for line in out.decode('utf-8').split('\n')]
    return output_lines

def get_dataset_infor(root="dataset/train"):
    root = Path(root)
    # get numbr of file
    audio_folder_num = {}
    for audio_folder in tqdm(root.iterdir(), desc="getting number of files"):
        audio_folder_num[audio_folder.name] = len(os.listdir(audio_folder))

    # get duration of files
    audio_folder_duration = {}
    for audio_folder in tqdm(root.iterdir(), desc="getting duration of files"):
        audio_folder_duration[audio_folder.name] = list([get_duration_file(audio_file) for audio_file in audio_folder.iterdir()])

    # get size of files
    audio_folder_size = {}
    for audio_folder in tqdm(root.iterdir(), desc="getting size of files"):
        audio_folder_size[audio_folder.name] = list([get_size_file(audio_file) for audio_file in audio_folder.iterdir()])
    
    # get amplitue of file take long time :v
#     audio_folder_amplitute = {}
#     for audio_folder in tqdm(root.iterdir(), desc="getting peak amplitude of files"):
#         audio_folder_amplitute[audio_folder.name] = list([get_amplitute_file(audio_file)[-1] for audio_file in audio_folder.iterdir()])
    
    return audio_folder_num, audio_folder_duration, audio_folder_size

def get_error_list(imposter_file):
    print("Get information from:", imposter_file)
    
    if os.path.isfile(imposter_file):
        with open(imposter_file, 'r') as rf:
            lines = [line.strip().replace('\n', '') for line in rf.readlines()]

        invalid_class = list(''.join(x.split(':')[1:]).strip() for x in filter(lambda x: True if ':' in x else False, lines))
        invalid_files = list(''.join(x.split('-')[1:]).strip() for x in filter(lambda x: True if '-' in x else False, lines))
        # len(invalid_files), len(invalid_class), invalid_class[-1], glob.glob("dataset/train/*").index(invalid_class[-1])

        invalid_details = {}
        for line in tqdm(lines):
            if ':' in line:
                k = ''.join(line.split(':')[1:]).strip()
                if k not in invalid_details:
                    invalid_details[k] = {}
            elif '.wav' in line:
                fp = ''.join(line.split(' - ')[1:]).strip()
                n = line.split('-')[0].strip().replace('[', '').replace(']', '').split('/')
                
                
                rate = float(n[0])/float(n[1])

                k = list(invalid_details.keys())[-1]
                invalid_details[k][fp] = rate

        return invalid_details
    else:
        return None
    
def export_dataset_details(root="dataset/train", save_dir="dataset/train_details/"):
    root = Path(root)
    print("Getting general information")
#     invalid_details = get_error_list('Imposter_callbot.txt')
    _, audio_folder_duration, audio_folder_size = get_dataset_infor(root)
    os.makedirs(save_dir, exist_ok=True)
    
    for audio_folder in tqdm(list(root.iterdir()), desc="Processing...", colour='red'):
        writefile = os.path.join(save_dir , f"{audio_folder.name}.csv")
        
        with open(writefile, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
                      'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
                      'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
                      'Crest factor', 'Flat factor', 'Peak count',
                      'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
                      'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
            
            spamwriter.writerow(header)
            
            for i, audio_file in enumerate(audio_folder.iterdir()):
                # general infor
                fp = str(Path(audio_folder, audio_file.name))
                duration = audio_folder_duration[audio_folder.name][i]
                size = audio_folder_size[audio_folder.name][i]
#                 # db = audio_folder_amplitute[audio_folder.name][i]
                
#                 if isinstance(invalid_details, dict):
#                     if str(root / audio_folder.name) in invalid_details.keys():
#                         if fp in invalid_details[str(root / audio_folder.name)]:
#                             error_rate = float(invalid_details[str(root / audio_folder.name)][fp])
#                         else:
#                             error_rate = 0
#                     else:
#                         error_rate = 0
#                 else:
#                     error_rate = 0
                error_rate = 0
                    
                # get full stats
                full_infor = list(get_audio_information_stats(fp)) # path
                details = {}
                condition = lambda x: 'Parsed_astats_0' in x
                filtered_lines = list(filter(condition, full_infor))
            
                for line in filtered_lines:
                    detail = line.replace(f"[{line.split('[')[-1].split(']')[0]}]", '').strip().split(':')
                    if detail[0] == 'Overall':
                        continue
                    details[detail[0]] = detail[1]
                
                row = [audio_file.name, duration, size/(1024), details['Min level'],details['Max level'],
                       details['Min difference'],details['Max difference'], details['Mean difference'],details['RMS difference'],
                       details['Peak level dB'],details['RMS level dB'], details['RMS peak dB'],details['RMS trough dB'],
                       details['Crest factor'],details['Flat factor'], details['Peak count'],
                       details['Noise floor dB'],details['Noise floor count'],details['Bit depth'],details['Dynamic range'],
                       details['Zero crossings'],details['Zero crossings rate'], error_rate, fp]

                spamwriter.writerow(row)
                
    return True

def update_dataset_details(root="dataset/train", save_dir="dataset/train_details/", error_file="Imposter_callbot2.txt"):
    root = Path(root)
    print("Getting general information")
    invalid_details = get_error_list(error_file)
    os.makedirs(save_dir, exist_ok=True)
    
    for audio_folder in tqdm(list(root.iterdir())[:], desc="Processing..."):
        reading_file = os.path.join(save_dir , f"{audio_folder.name}.csv")
        writing_file = reading_file
        
        rows = []
        with open(reading_file, 'r', newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                rows.append(row)
        
        with open(writing_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
                      'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
                      'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
                      'Crest factor', 'Flat factor', 'Peak count',
                      'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
                      'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
            
            spamwriter.writerow(header)
            
            for i, audio_file in enumerate(list(audio_folder.iterdir())):
                fp = rows[i][-1]
                
                if isinstance(invalid_details, dict):
                    if str(root / audio_folder.name) in invalid_details.keys():
                        if fp in invalid_details[str(root / audio_folder.name)]:
                            error_rate = float(invalid_details[str(root / audio_folder.name)][fp])
                        else:
                            error_rate = 0
                    else:
                        error_rate = 0
                else:
                    error_rate = 0
                    
            
                row_new = rows[i]
                row_new[-2] = error_rate
                spamwriter.writerow(row_new)
                
    return True
        
def read_blacklist(id, duration_limit=1.0, dB_limit=-16, error_limit=0, noise_limit=-15):
    blacklist = []
    readfile = str(Path("dataset/train_details/", f"{id}.csv"))
    
    with open(readfile, 'r', newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        #             header = ['File name', 'Duration', 'Size(MB)', 'Min level', 'Max level', 
        #                       'Min difference', 'Max difference', 'Mean difference', 'RMS difference', 
        #                       'Peak level dB', 'RMS level dB',   'RMS peak dB', 'RMS trough dB', 
        #                       'Crest factor', 'Flat factor', 'Peak count',
        #                       'Noise floor dB', 'Noise floor count', 'Bit depth', 'Dynamic range', 
        #                       'Zero crossings', 'Zero crossings rate', 'Error rate', 'Full path']
        for row in spamreader:
            
            short = (float(row[1]) < duration_limit)
            low_amp = (float(row[9]) < dB_limit)
            large_err = (float(row[-2]) > error_limit)
            noise = (float(row[17]) > noise_limit)
            
            if  short or low_amp or large_err or noise:
                blacklist.append(Path(row[-1]))
                
    return list(set(blacklist))
        
if __name__ == "__main__":
    export_dataset_details(root="dataset/test_callbot/public", save_dir="dataset/details/test_cb_public")
#     update_dataset_details(root="dataset/train", save_dir="dataset/train_details_full/", error_file="Imposter_v2.txt")
    