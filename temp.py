
import csv
import glob
import os

import pandas as pd
from torchsummary import summary

from models.RepVGG import *


# from models.ResNetSE34V2 import *
def calcualte_similarity_of_test_result():
    test = pd.read_csv(
        "exp/dump/submission_list_test11091050.csv")  # 86.55%

    ref = pd.read_csv(
        "exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")  # 87.38%
    # similarity accepted: 74% -> 100%
    data1 = list(ref['label'])
    data2 = list(test['label'])

    similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
    print(similarity.count(1))
    print(f"similarity: {similarity.count(1)/len(data1)}")


model = MainModel(model='RepVGG-B0')
summary(model, (1, 64, 400), batch_size=128)
