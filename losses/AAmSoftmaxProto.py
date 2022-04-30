#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

import losses.AAmSoftmax as aamsoftmax
import losses.AngularProto as angleproto


class AAmSoftmaxProto(nn.Module):

    def __init__(self, **kwargs):
        super(AAmSoftmaxProto, self).__init__()

        self.test_normalize = True

        self.aamsoftmax = aamsoftmax.AAmSoftmax(**kwargs)
        self.angleproto = angleproto.AngularProto(**kwargs)

        print('Initialised SoftmaxPrototypical Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.aamsoftmax(x, label)

        nlossP, _ = self.angleproto(x, label)

        return nlossS+nlossP, prec1