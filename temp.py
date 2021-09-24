
import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchsummary import summary
from tqdm.auto import tqdm

from models.RepVGG import *
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


# calcualte_similarity_of_test_result()
# pdist = torch.nn.PairwiseDistance(p=2)
# in1 = torch.randn(10, 2).cpu().numpy()
# in2 = torch.randn(10, 2).cpu().numpy()
# print(in1)
# print(in2)
# # output = abs(1-cosine(in1, in2))
# output = F.cosine_similarity(torch.tensor(in1).float(), torch.tensor(in2).float(), dim=1)
# # # output = F.normalize(output, p=2, dim=1).detach().numpy()
# print(np.mean(abs(output).cpu().numpy()))
# x = -1 * np.mean([output])
# print(x)


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


# overwirte()
# print(MainModel())
summary(MainModel(model='RepVGG-D2se'), (16240,), 2)
