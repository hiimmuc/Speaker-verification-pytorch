# import torch
# import torch.nn.functional as F
# import time
# x = torch.randn((256, 96, 128, 128)).cuda()
# t = time.time()
# x = F.avg_pool2d(x, x.size()[2:])

# %timeit F.adaptive_avg_pool2d(x, (1, 1))

# %timeit torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
# from utils import *

# path = './dataset/wavs/272-M-26/speaker_272-10.wav'
# audio = loadWAV(path, 100, evalmode=False)
# print(audio.shape)
import csv
import glob
import os

import pandas as pd

# path1 = ''
# path2 = ''

csv1 = pd.read_csv(
    "exp/dump/submission_list_test2.csv")  # 86.55%
csv2 = pd.read_csv(
    "exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")  # 87.38%

data1 = list(csv1['label'])
data2 = list(csv2['label'])
# print(data1)
# print(data2)
similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
print(similarity.count(1))
print(f"similarity: {similarity.count(1)/len(data1)}")
# path = 'dataset/val.txt'
# with open(path, 'r') as f:
#     lines = f.readlines()
#     # for line in lines:
#     #     print(line)
#     #     os.system(f"rm dataset/{line.split(' ')[0]}")
# raw_data = list(filter(lambda x: 'augment' not in x, lines))
# augment_data = list(filter(lambda x: 'augment' in x, lines))
# print(len(raw_data), len(augment_data))
