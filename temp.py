
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.common_types import T

from utils import *

# from torchvision import transforms


# from models.ResNetSE34V2 import *
def calcualte_similarity_of_test_result():
    test = pd.read_csv(
        "exp\dump\submission_backbone_softmax_normthres.csv")  # 86.55%

    ref = pd.read_csv(
        "exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")  # 87.38%
    # similarity accepted: 74% -> 100%
    data1 = list(ref['label'])
    data2 = list(test['label'])

    similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
    print(similarity.count(1))
    print(f"similarity: {similarity.count(1)/len(data1)}")


calcualte_similarity_of_test_result()
pdist = torch.nn.PairwiseDistance(p=2)
in1 = torch.randn(1, 512)
in2 = torch.randn(1, 512)
print(in1)
print(in2)
output = F.cosine_similarity(in1, in2)
# output = F.normalize(output, p=2, dim=1).detach().numpy()
print(output)
# x = -1 * np.mean([output])
# print(x)

# ref = pd.read_csv(
#     "exp/dump/submission_backbone_amsoftmax.csv")  # 87.38%
# # similarity accepted: 74% -> 100%
# data1 = list(ref['score'])
# data1 = [float(i) for i in data1]
# data1 = [float(i)/max(data1) for i in data1]
# print(data1[:20])
