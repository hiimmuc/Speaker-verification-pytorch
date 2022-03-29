'''
From https://github.com/cvqluu/GE2E-Loss
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from utils import accuracy
from .GE2ELossV2 import GE2ELossV2

class GE2ELoss_fusion(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
        super(GE2ELoss_fusion, self).__init__()
        self.test_normalize = True
        
        self.ge2e_softmax = GE2ELossV2(init_w=10.0, init_b=-5.0, loss_method='softmax')
        self.ge2e_contrast = GE2ELossV2(init_w=10.0, init_b=-5.0, loss_method='contrast')

    def forward(self, dvecs, label=None):
        '''
        Calculates the GE2E loss for an input of dimensions (num_speakers, num_utts_per_speaker, dvec_feats)
        num_utts_per_speaker >= 2, in paper 10
        '''
        assert dvecs.size()[1] >= 2

        loss_softmax, prec1 = self.ge2e_softmax(dvecs, label)
        loss_contrast, _ = self.ge2e_contrast(dvecs, label)
        
        return loss_softmax + loss_contrast, prec1
