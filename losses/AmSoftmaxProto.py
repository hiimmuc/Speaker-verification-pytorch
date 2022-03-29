#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.AmSoftmax as amsoftmax
import losses.AngularProto as angleproto


class AmSoftmaxProto(nn.Module):

    def __init__(self, **kwargs):
        super(AmSoftmaxProto, self).__init__()

        self.test_normalize = True

        self.amsoftmax = amsoftmax.AmSoftmax(**kwargs)
        self.angleproto = angleproto.AngularProto(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.amsoftmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.angleproto(x, None)

        return 1.0 * nlossS + 1.0 * nlossP, prec1
