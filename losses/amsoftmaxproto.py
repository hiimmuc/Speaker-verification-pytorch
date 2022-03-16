#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.amsoftmax as amsoftmax
import losses.angleproto as angleproto


class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.amsoftmax = amsoftmax.LossFunction(**kwargs)
        self.angleproto = angleproto.LossFunction(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.amsoftmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.angleproto(x, label)

        return 1.0 * nlossS + 1.0 * nlossP, prec1
