
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from numpy.core.fromnumeric import argmin
from torchsummary import summary
from tqdm.auto import tqdm

from models.RawNetv2 import *
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


def check_inequality_data():
    '''to check the inequality of dataset from wavs folder
    '''
    root = Path("dataset/wavs")
    audio_folder_num = {}
    for audio_folder in root.iterdir():
        audio_folder_num[audio_folder.name] = len(os.listdir(audio_folder))
    mean_num = np.mean(list(audio_folder_num.values()))
    print("Total files:", sum(audio_folder_num.values()))
    print("mean number of audio files each folder", mean_num)
    print("min and max of number of files:", min(audio_folder_num.values()), max(audio_folder_num.values()))
    print(list(audio_folder_num.keys())[argmin(audio_folder_num.values())])

    greater_than_mean = [k for k, v in audio_folder_num.items() if v > 1.2 * mean_num]
    lower_than_mean = [k for k, v in audio_folder_num.items() if v < 0.9 * mean_num]

    print('//===================================')
    # check for the total duration of each folder
    audio_folder_duration = {}
    for audio_folder in root.iterdir():
        audio_folder_duration[audio_folder.name] = sum([audio_file.stat().st_size for audio_file in audio_folder.iterdir()])
    mean_duration = np.mean(list(audio_folder_duration.values()))
    print("Total size:", sum(audio_folder_duration.values())/1e6, "MB")
    print("mean duration of audio files each folder", mean_duration/(1024*1024), "MB")
    print("min and max of duration:", min(audio_folder_duration.values())/(1024*1024), max(audio_folder_duration.values())/(1024*1024), "MB")
    print(list(audio_folder_duration.keys())[argmin(audio_folder_duration.values())])

    greater_than_mean_duration = [k for k, v in audio_folder_duration.items() if v > 1.2 * mean_duration]
    lower_than_mean_duration = [k for k, v in audio_folder_duration.items() if v < 0.9 * mean_duration]

    print('//===================================')

    common_long = np.intersect1d(greater_than_mean, greater_than_mean_duration)
    common_short = np.intersect1d(lower_than_mean, lower_than_mean_duration)

    print("greater than mean in number of files:", len(greater_than_mean))
    print("lower than mean in number of files:", len(lower_than_mean))
    print("greater than mean in duration:", len(greater_than_mean_duration))
    print("lower than mean in duration:", len(lower_than_mean_duration))
    print("common_long:", len(common_long))
    print("common_short:", len(common_short))

    with open("data_inequality.txt", 'w') as f:
        f.write("Long folder: \n")
        for line in common_long:
            f.write(line + '\n')

        f.write("Short folder: \n")
        for line in common_short:
            f.write(line + '\n')


def create_full_augmented_dataset():
    root = Path("dataset/aug_wavs")
    for audio_folder in tqdm(root.iterdir()):
        f = glob.glob(str(audio_folder) + "/*.wav")
        non_aug = list(filter(lambda x: "augmented" not in x, f))
        # print(non_aug[0])
        for file in non_aug:
            os.remove(file)


if __name__ == '__main__':
    # check_inequality_data()
    # create_full_augmented_dataset()
    # check_result("exp\dump\submission_list_test_1310_ambase.csv")

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))
    # audio =
    summary(model, (16240, ), batch_size=128)
