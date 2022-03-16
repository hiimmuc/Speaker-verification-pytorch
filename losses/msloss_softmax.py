#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.msloss as msloss
import losses.softmax as softmax


class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.LossFunction(**kwargs)
        self.msloss = msloss.LossFunction(**kwargs)

        print('Initialised Multi Similarity softmax Loss')

    def forward(self, x, label=None):

        

        nlossS, prec1 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(x.size()[1]))

        nlossP, _ = self.msloss(x, label)

        return nlossS+nlossP, prec1
