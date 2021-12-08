#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from models.ResNetBlocks import *


class x_vector_model(nn.Module):

    def __init__(self, numSpkrs=5994, p_dropout=0.05):
        super(x_vector_model, self).__init__()
        self.tdnn1 = nn.Conv1d(in_channels=257, out_channels=512, kernel_size=5, dilation=1)
        self.bn_tdnn1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn1 = nn.Dropout(p=p_dropout)

        self.tdnn2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, dilation=2)
        self.bn_tdnn2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn2 = nn.Dropout(p=p_dropout)

        self.tdnn3 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=7, dilation=3)
        self.bn_tdnn3 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn3 = nn.Dropout(p=p_dropout)

        self.tdnn4 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=1, dilation=1)
        self.bn_tdnn4 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_tdnn4 = nn.Dropout(p=p_dropout)

        self.tdnn5 = nn.Conv1d(in_channels=512, out_channels=1500, kernel_size=1, dilation=1)
        self.bn_tdnn5 = nn.BatchNorm1d(1500, momentum=0.1, affine=False)
        self.dropout_tdnn5 = nn.Dropout(p=p_dropout)

        self.fc1 = nn.Linear(3000,512)
        self.bn_fc1 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc1 = nn.Dropout(p=p_dropout)

        self.fc2 = nn.Linear(512,512)
        self.bn_fc2 = nn.BatchNorm1d(512, momentum=0.1, affine=False)
        self.dropout_fc2 = nn.Dropout(p=p_dropout)

        self.fc3 = nn.Linear(512,numSpkrs)
        
        self.instancenorm = nn.InstanceNorm1d(257)

    def forward(self, x, eps=0.01):
        # Note: x must be (batch_size, feat_dim, chunk_len)

        stft = torch.stft(x, 512, hop_length=int(0.01*16000), win_length=int(0.025*16000), window=torch.hann_window(int(0.025*16000)), center=False, normalized=False, onesided=True)
        stft = (stft[:,:,:,0].pow(2)+stft[:,:,:,1].pow(2)).pow(0.5)

        x = self.instancenorm(stft).detach()
        
        #print(x.shape)
        
        x = self.dropout_tdnn1(self.bn_tdnn1(F.relu(self.tdnn1(x))))
        #print(x.shape)
        x = self.dropout_tdnn2(self.bn_tdnn2(F.relu(self.tdnn2(x))))
        #print(x.shape)
        x = self.dropout_tdnn3(self.bn_tdnn3(F.relu(self.tdnn3(x))))
        #print(x.shape)
        x = self.dropout_tdnn4(self.bn_tdnn4(F.relu(self.tdnn4(x))))
        #print(x.shape)
        x = self.dropout_tdnn5(self.bn_tdnn5(F.relu(self.tdnn5(x))))
        #print(x.shape)

        if self.training:
            shape=x.size()
            noise = torch.cuda.FloatTensor(shape) if torch.cuda.is_available() else torch.FloatTensor(shape)
            torch.randn(shape, out=noise)
            x += noise*eps

        stats = torch.cat((x.mean(dim=2), x.std(dim=2)), dim=1)
        #print(x.mean(dim=2).shape)
        #print(stats.shape)
        x = self.dropout_fc1(self.bn_fc1(F.relu(self.fc1(stats))))
        #print(x.shape)
        x = self.dropout_fc2(self.bn_fc2(F.relu(self.fc2(x))))
        #print(x.shape)
        #x = self.fc3(x)
        return x


def x_vector(nOut=512, **kwargs):
    # Number of filters

    model = x_vector_model()
    return model
