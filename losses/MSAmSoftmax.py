#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.MultiSimilarity_v2 as msloss
import losses.ARmSoftmax as armsoftmax


class MSAmSoftmax(nn.Module):

    def __init__(self, **kwargs):
        super(MSAmSoftmax, self).__init__()

        self.test_normalize = True

        self.armsoftmax = armsoftmax.ARmSoftmax(**kwargs)
        self.msloss = msloss.MultiSimilarity_v2(**kwargs)

        print('Initialised Multi Similarity V2 Loss')

    def forward(self, x, label=None):
        weight = 0.6
        nlossCE, prec1 = self.armsoftmax(x, label)

        nlossML, _ = self.msloss(x, label)

        return (1-weight) * nlossCE + weight * nlossML, prec1
