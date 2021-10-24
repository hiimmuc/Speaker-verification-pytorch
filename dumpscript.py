
import csv
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from numpy.core.fromnumeric import argmin
from torchsummary import summary
from tqdm.auto import tqdm

from models.ResNetSE34V2 import *
from utils import *


def calcualte_similarity_of_test_result():
    test_path = Path("exp/dump/submission_backbone_softmax_normthres.csv")  # 86.55%

    ref_path = Path("exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")

    test = pd.read_csv(test_path)  # 86.55%

    ref = pd.read_csv(ref_path)  # 87.38%

    # similarity accepted: 74% -> 100%
    data1 = list(ref['label'])
    data2 = list(test['label'])

    similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
    print(similarity.count(1))
    print(f"similarity: {similarity.count(1)/len(data1)}")


def overwirte():
    """fix same name"""
    test = Path("exp/dump/submission_backbone_softmax_normthres.csv")  # 86.55%

    ref = Path("exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")
    lines = []
    scores = []
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            lines.append([row[0], row[1]])
    with open(test, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            scores.append([row[2], row[3]])
    new_lines = [lines[i] + scores[i] for i in range(len(lines))]
    with open(test, 'w', newline='') as wf:
        spamwriter = csv.writer(wf, delimiter=',')
        spamwriter.writerow(['audio_1', 'audio_2', 'label', 'score'])
        for line in tqdm(new_lines):
            spamwriter.writerow(line)
    print('Done!')
    pass


def check_result(answer):
    ref = Path("exp\data_truth.txt")
    com = Path(answer)

    reference = {}
    comparation = {}
    #  read from reference csv file
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=' ')
        # next(spamreader, None)
        for row in tqdm(spamreader, desc="read reference"):
            # print(row[1:])
            key = f"{row[1]}/{row[2]}"
            reference[key] = row[0]

    #  read from compare csv file
    with open(com, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in tqdm(spamreader, desc="read compare"):
            key = f"{row[0]}/{row[1]}"
            comparation[key] = row[2]

    assert len(reference) == len(comparation) or len(reference) == len(comparation) + 1, "length of reference and comparation is not equal"

    # precision
    count = 0
    for k, v in comparation.items():
        if reference[k] == v:
            count += 1
        pass
    print(count/len(comparation), '%')


if __name__ == '__main__':
    # check_inequality_data()
    # create_full_augmented_dataset()
    # check_result("exp\dump\submission_list_test_1310_ambase.csv")

    model = MainModel(preprocess=True)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))
    # audio =
    summary(model, (1, 64, 102), batch_size=128)

    # audio_path = r"dataset\public_test\data_test\0a0056eec2d8de0b89e52849b1c2844e.wav"
    # mels = librosa.feature.melspectrogram(y=sf.read(audio_path)[0], sr=16000, n_fft=512, hop_length=160, win_length=400, n_mels=64)
    # print(mels.shape)
