
from models.RepVGG import *
import csv
import glob
import os

import pandas as pd

csv1 = pd.read_csv(
    "exp/dump/submission_list_test11091050.csv")  # 86.55%

csv2 = pd.read_csv(
    "exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")  # 87.38%
# similarity accepted: 74% -> 100%
data1 = list(csv1['label'])
data2 = list(csv2['label'])

similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
print(similarity.count(1))
print(f"similarity: {similarity.count(1)/len(data1)}")

# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.


# from ret_benchmark.losses.registry import LOSS


# @LOSS.register('ms_loss')
# class LossFunction(nn.Module):
#     def __init__(self, margin=0.1, scale_neg=50, scale_pos=2, ** kwargs):
#         super(LossFunction, self).__init__()
#         self.thresh = 0.5
#         self.margin = margin
#         self.scale_pos = scale_pos
#         self.scale_neg = scale_neg

#     def forward(self, feats, labels):
#         assert feats.size(0) == labels.size(0), \
#             f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
#         batch_size = feats.size(0)
#         # feat: batch_size x outdim
#         feats = F.normalize(feats, dim=1)
#         sim_mat = torch.matmul(feats, torch.t(feats))

#         epsilon = 0.1
#         loss = list()
#         c = 0

#         for i in range(batch_size):
#             # mining step same as hard mining loss  https://github.com/bnu-wangxun/Deep_Metric/blob/master/losses/HardMining.py
#             pos_pair_ = torch.masked_select(sim_mat[i], labels == labels[i])
#             #  move itself
#             pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)

#             neg_pair_ = torch.masked_select(sim_mat[i], labels != labels[i])

#             neg_pair = torch.masked_select(
#                 neg_pair_, neg_pair_ > min(pos_pair_) - self.margin)
#             pos_pair = torch.masked_select(
#                 pos_pair_, pos_pair_ < max(neg_pair_) + self.margin)

#             if len(neg_pair) < 1 or len(pos_pair) < 1:
#                 c += 1
#                 continue

#             # weighting step
#             pos_loss = 1.0 / self.scale_pos * torch.log(
#                 1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
#             neg_loss = 1.0 / self.scale_neg * torch.log(
#                 1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))

#             loss.append(pos_loss + neg_loss)

#         if len(loss) == 0:
#             return torch.zeros([], requires_grad=True)

#         loss = sum(loss) / batch_size
#         prec1 = float(c) / batch_size
#         return loss, prec1


# if __name__ == '__main__':
#     loss = LossFunction()
#     feats = torch.rand([64, 512])
#     labels = torch.rand(64)
#     loss, prec1 = loss(feats, labels)
#     print(loss)
#     print(prec1)

print(MainModel())
