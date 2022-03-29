#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.MultiSimilarity_v2 as msloss
import losses.AmSoftmax as amsoftmax


class MSAmSoftmax(nn.Module):

    def __init__(self, **kwargs):
        super(MSAmSoftmax, self).__init__()

        self.test_normalize = True

        self.amsoftmax = amsoftmax.AmSoftmax(**kwargs)
        self.msloss = msloss.MultiSimilarity_v2(**kwargs)

        print('Initialised Multi Similarity softmax Loss')

    def forward(self, x, label=None):
        weight = 0.6
        nlossCE, prec1 = self.amsoftmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(x.size()[1]))

        nlossML, _ = self.msloss(x, label)

        return (1-weight) * nlossCE + weight * nlossML, prec1
