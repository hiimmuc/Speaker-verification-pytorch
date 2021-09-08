# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
from config import *
from torch import nn
from utils import *

# from ret_benchmark.losses.registry import LOSS


# @LOSS.register('ms_loss')
class LossFunction(nn.Module):
    def __init__(self, margin=0.1, scale_neg=50, scale_pos=2, ** kwargs):
        super(LossFunction, self).__init__()
        self.thresh = 0.5
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        # feat: batch_size x outdim
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()
        c = 0
        for i in range(batch_size):
            # mining step same as hard mining loss  https://github.com/bnu-wangxun/Deep_Metric/blob/master/losses/HardMining.py
            pos_pair_ = torch.masked_select(sim_mat[i], labels == labels[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat[i], labels != labels[i])

            neg_pair = torch.masked_select(
                neg_pair_, neg_pair_ > min(pos_pair_) - self.margin)
            pos_pair = torch.masked_select(
                pos_pair_, pos_pair_ < max(neg_pair_) + self.margin)

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                c += 1
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))

            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        prec1 = float(c) / batch_size
        return loss, prec1
