#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 17:32:03 2020
@author: krishna
"""

import torch
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from utils import PreEmphasis

# TODO: add option block, layers, filter, ... to block
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes3, planes, stride=1, downsample=None):
        super(BasicBlock3x3, self).__init__()
        self.conv1 = conv3x3(inplanes3, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)

        return out


class RawNet(nn.Module):
    def __init__(self, input_channel=1, nOut=512, n_mels=64, **kwargs):
        self.inplanes3 = 128
        super(RawNet, self).__init__()
        self.conv1 = nn.Conv1d(input_channel, 128, kernel_size=3, stride=3, padding=0,
                               bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        #############################################################################

        self.resblock_1_1 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.resblock_1_2 = self._make_layer3(BasicBlock3x3, 128, 1, stride=1)
        self.maxpool_resblock_1 = nn.MaxPool1d(
            kernel_size=3, stride=3, padding=0)
        #############################################################################
        self.resblock_2_1 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_2 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_3 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.resblock_2_4 = self._make_layer3(BasicBlock3x3, 256, 1, stride=1)
        self.maxpool_resblock_2 = nn.MaxPool1d(
            kernel_size=3, stride=3, padding=0)

        ############################################################################
        self.gru = nn.GRU(input_size=256, hidden_size=1024,
                          dropout=0.2, bidirectional=False, batch_first=True)
        self.spk_emb = nn.Linear(1024, 128)
        # self.drop = nn.Dropout(p=0.2)
        self.output_layer = nn.Linear(128, nOut)
        # ##########################################################################


    def _make_layer3(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes3 != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes3, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes3, planes, stride, downsample))
        self.inplanes3 = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes3, planes))

        return nn.Sequential(*layers)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)

        # ResBlock-1
        out = self.resblock_1_1(out)
        out = self.maxpool_resblock_1(out)
        out = self.resblock_1_2(out)
        out = self.maxpool_resblock_1(out)
        # Resblock-2
        out = self.resblock_2_1(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_2(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_3(out)
        out = self.maxpool_resblock_2(out)
        out = self.resblock_2_4(out)
        out = self.maxpool_resblock_2(out)
        # GRU
        out = out.permute(0, 2, 1)
        out, _ = self.gru(out)
        out = out.permute(0, 2, 1)
        spk_embeddings = self.spk_emb(out[:, :, -1])
        preds = self.output_layer(spk_embeddings)

        return preds

def MainModel(nOut=512, **kwargs):
    model = RawNet(nOut=nOut)
    return model
