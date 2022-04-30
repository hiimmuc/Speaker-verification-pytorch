#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import pdb
import time

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy


class AmSoftmax(nn.Module):
    def __init__(self, nOut, nClasses, margin=0.1, scale=30, **kwargs):
        super(AmSoftmax, self).__init__()

        self.test_normalize = True

        self.m = margin
        self.s = scale
        self.in_feats = nOut
        self.W = torch.nn.Parameter(torch.randn(
            nOut, nClasses), requires_grad=True)
        self.ce = nn.CrossEntropyLoss()
        nn.init.xavier_normal_(self.W, gain=1)

        print('Initialised AMSoftmax m=%.3f s=%.3f' % (self.m, self.s))

    def forward(self, x, label=None):
        if len(x.shape) == 3:
            label = label.repeat_interleave(x.shape[1])
            x = x.reshape(-1, self.in_feats)
        elif len(x.shape) == 2:
            pass
        else:
            raise "Invalid shape of input"
        assert x.size()[0] == label.size()[0]
        assert x.size()[1] == self.in_feats

        device = x.get_device()

        x_norm = torch.norm(x, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x, x_norm)
        w_norm = torch.norm(self.W, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm = torch.div(self.W, w_norm)
        costh = torch.mm(x_norm, w_norm) # costh: cos theta = W.T * f // (||W.T||.||f||)

        label_view = label.view(-1, 1)
        if label_view.is_cuda:
            label_view = label_view.cpu()

        delt_costh = torch.zeros(costh.size()).scatter_(1, label_view, self.m)
        if x.is_cuda:
            delt_costh = delt_costh.cuda(device=device)
        
        costh_m = costh - delt_costh
        costh_m_s = self.s * costh_m
        
        loss = self.ce(costh_m_s, label)
        prec1 = accuracy(costh_m_s.detach(), label.detach(), topk=(1,))[0]
        return loss, prec1
