# Copyright (c) Malong Technologies Co., Ltd.
# All rights reserved.
#
# Contact: github@malong.com
#
# This source code is licensed under the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


class MultiSimilarity(nn.Module):
    def __init__(self, margin=0.1, scale_neg=50.0, scale_pos=2.0, ** kwargs):
        super(MultiSimilarity, self).__init__()
        self.test_normalize = True
        self.thresh = 0.5  # lambda
        self.margin = margin
        self.scale_pos = scale_pos  # alpha
        self.scale_neg = scale_neg  # beta

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        # feats: batch x n x d
        # batchxn -> batch.n n: nPerSpeaker ; size [64] -> [640], repeat by number of utterances
        labels = labels.repeat_interleave(feats.size()[1])
        # batchxnxd -> batch.n x d; [64, 5, 128] -> [320, 128]
        feats = feats.reshape(-1, feats.size()[-1])
        
        batch_size = feats.size(0)
        # feat: batch_size x outdim

        sim_mat = torch.matmul(feats, torch.t(feats))
        # test: normalize sim mat so all values is below 1
        sim_mat_norm = F.normalize(sim_mat)

        epsilon = 1e-5
        losses = list()
        c = 0

        for i in range(batch_size):
            # pair mining step
            # implement same as hard mining loss  https://github.com/bnu-wangxun/Deep_Metric/blob/master/losses/HardMining.py
            pos_pair_ = torch.masked_select(sim_mat_norm[i], labels == labels[i])

            #  move itself
            pos_pair_ = torch.masked_select(pos_pair_, pos_pair_ < 1 - epsilon)
            neg_pair_ = torch.masked_select(sim_mat_norm[i], labels != labels[i])

            neg_pair = torch.masked_select(
                neg_pair_, neg_pair_ > min(pos_pair_) - self.margin)
            pos_pair = torch.masked_select(
                pos_pair_, pos_pair_ < max(neg_pair_) + self.margin)

            if len(neg_pair) < 1:
                c += 1
                continue
            if len(pos_pair) < 1:
                continue

            # pair weighting stage
            pos_loss = (1.0 / self.scale_pos) * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = (1.0 / self.scale_neg) * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss_ms = pos_loss + neg_loss

            losses.append(loss_ms)

        if len(losses) == 0:
            return torch.zeros([], requires_grad=True), 0

        loss = sum(losses) / batch_size
        
        prec1 = float(c) * 100 / batch_size
        prec_sim = accuracy(sim_mat.view(-1, batch_size).detach(), labels.detach(), topk=(1,))[0]
        
        return loss, prec_sim


if __name__ == '__main__':
    loss = LossFunction()
    feats = torch.rand([64, 512])
    labels = torch.rand(64)
    loss, pred = loss(feats, labels)
    print(loss)
    print(pred)
