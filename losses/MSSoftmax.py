#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.MultiSimilarity as msloss
import losses.Softmax as softmax


class MSSofmaxLoss(nn.Module):

    def __init__(self, **kwargs):
        super(MSSofmaxLoss, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.Softmax(**kwargs)
        self.msloss = msloss.MultiSimilarity(**kwargs)

        print('Initialised Multi Similarity softmax Loss')

    def forward(self, x, label=None):

        

        nlossS, prec1 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(x.size()[1]))

        nlossP, _ = self.msloss(x, label)

        return nlossS+nlossP, prec1
