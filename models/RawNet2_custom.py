import math
from collections import OrderedDict

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from numpy.core.fromnumeric import transpose
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data
from torchsummary import summary
from utils import *
from models.RawNet2_baseline import *

class RawNet2(nn.Module):
    """
    Refactored RawNet2 architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    """

    def __init__(self, block, layers, nb_filters,
                 audio_spec,
                 front_proc='sinc',  aggregate='sap',
                 att_dim=128,
                 code_dim=512,
                 in_channels=1,
                 log_input=True,
                 nb_gru_layers=1, 
                 gru_node=1024,
                 first_conv_size=251,
                 **kwargs):
        super(RawNet2, self).__init__()

        self.inplanes = nb_filters[0]
        self.log_input = log_input
    
        #####
        # first layers before residual blocks
        #####
        self.front_proc = front_proc
        if self.front_proc == 'conv':
            self.conv1 = nn.Conv1d(
                in_channels,
                nb_filters[0],
                kernel_size=3,
                stride=3,
                padding=0        
            )
        elif self.front_proc == 'sinc':
            #  sinc layer
            sample_rate = audio_spec['sample_rate']
            hoplength = int(audio_spec['hop_len'] * sample_rate * 1e-3)
            winlength = int(audio_spec['win_len'] * sample_rate * 1e-3)
            nb_samp = int(audio_spec['sentence_len'] * sample_rate)

            self.ln = LayerNorm(nb_samp)
            self.first_conv = SincConv_fast(in_channels=in_channels,
                                            out_channels=nb_filters[0],
                                            kernel_size=first_conv_size,
                                            sample_rate=int(sample_rate),
                                            stride=1, padding=0, dilation=1, 
                                            bias=False, groups=1, 
                                            min_low_hz=50, min_band_hz=50)

            self.first_bn = nn.BatchNorm1d(nb_filters[0])

        #####
        # residual blocks for frame-level representations
        #####
        self.layer1 = self._make_layer(block, nb_filters[0], layers[0])
        self.layer2 = self._make_layer(block, nb_filters[1], layers[1])
        self.layer3 = self._make_layer(block, nb_filters[2], layers[2])
        self.layer4 = self._make_layer(block, nb_filters[3], layers[3])
        self.layer5 = self._make_layer(block, nb_filters[4], layers[4])
        self.layer6 = self._make_layer(block, nb_filters[5], layers[5])
        
        #####
        ## Aggregation layer
        #####
        self.aggregate = aggregate
        if self.aggregate == 'gru':
            # gru mode
            self.bn_before_gru = nn.BatchNorm1d(nb_filters[5])

            self.gru = nn.GRU(input_size=nb_filters[5],
                              hidden_size=gru_node,
                              num_layers=nb_gru_layers,
                              batch_first=True)

            self.fc_after_gru = nn.Linear(in_features=gru_node,
                                          out_features=code_dim)
            
        elif self.aggregate == 'sap':
            # SAP: statistic attentive pooling
            self.bn_before_agg = nn.BatchNorm1d(nb_filters[5])
            self.attention_sap = Classic_Attention(nb_filters[5], nb_filters[5])           
            
        else:
            # aggregate to utterance(segment)-level asp
            self.bn_before_agg = nn.BatchNorm1d(nb_filters[5])
            self.attention = nn.Sequential(
                nn.Conv1d(nb_filters[5], att_dim, kernel_size=1),
                nn.LeakyReLU(),
                nn.BatchNorm1d(att_dim),
                nn.Conv1d(att_dim, nb_filters[5], kernel_size=1),
                nn.Softmax(dim=-1),
            )       
  
        #####
        # speaker embedding layer
        #####

        self.fc = nn.Linear(nb_filters[5] * 2, code_dim)
        # self.fc2 = nn.Linear(code_dim , code_dim)
        self.lrelu = nn.LeakyReLU(0.3)  # keras style

        #####
        # initialize
        #####
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.sig = nn.Sigmoid()


    def _make_layer(self, block, planes, nb_layer, downsample_all=False):
        if downsample_all:
            downsamples = [True] * (nb_layer)
        else:
            downsamples = [False] * (nb_layer - 1) + [True]
        layers = []
        for d in downsamples:
            layers.append(block(self.inplanes, planes, downsample=d))
            self.inplanes = planes
        return nn.Sequential(*layers)
    
    def weighted_sd(self, inputs, attention_weights, mean):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        hadmard_prod = torch.mul(inputs,el_mat_prod)
        variance = torch.sum(hadmard_prod,1) - torch.mul(mean,mean)
        return variance
    
    
    def stat_attn_pool(self,inputs,attention_weights):
        el_mat_prod = torch.mul(inputs,attention_weights.unsqueeze(2).expand(-1,-1,inputs.shape[-1]))
        mean = torch.mean(el_mat_prod,1)
        variance = self.weighted_sd(inputs,attention_weights,mean)
        stat_pooling = torch.cat((mean,variance),1)
        return stat_pooling

    def forward(self, x):
        #####
        # first layers before residual blocks
        #####
        
        if self.front_proc == 'conv':
            x = x.unsqueeze(1)
            # conv
            x = self.conv1(x)
        elif self.front_proc == 'sinc':
            x = self.ln(x)           
            x = x.unsqueeze(1)
            # sinc
            x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
            x = self.first_bn(x)
            x = self.lrelu(x)
        
        #####
        # frame-level
        #####
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        
        if self.aggregate == 'gru':
            #####
            # gru
            #####
            x = self.bn_before_gru(x)
            x = self.lrelu(x)
            x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
            self.gru.flatten_parameters()
            x, _ = self.gru(x)
            x = x[:, -1, :]
            
            x = self.fc_after_gru(x) # speaker embedding layer
            
        elif self.aggregate == 'sap':
            x = self.bn_before_agg(x)
            x = self.lrelu(x)
            x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
            w = self.attention_sap(x)
            x = self.stat_attn_pool(x, w)
            
            x = self.fc(x)
            # x = self.fc2(x)           
            
        else:
            assert self.aggregate == 'asp'
            #####
            # aggregation: attentive statistical pooling
            #####
            x = self.bn_before_agg(x)
            x = self.lrelu(x)
            w = self.attention(x)
            m = torch.sum(x * w, dim=-1)
            s = torch.sqrt(
                (torch.sum((x ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
            x = torch.cat([m, s], dim=1)
            x = x.view(x.size(0), -1)
            
            x = self.fc(x) # speaker embedding layer        

        x = x.squeeze() 
        return x


def MainModel(nOut=512, **kwargs):
    layers = [1, 1, 1, 2, 1, 2]
    nb_filters = [128, 128, 256, 256, 512, 512]
    # layers = [1, 1, 3, 4, 6, 3]
    # nb_filters = [128, 128, 256, 256, 256, 256]

    model = RawNet2(
        RawNetBasicBlock,
        layers=layers,
        nb_filters=nb_filters,
        code_dim=nOut,
        **kwargs
    )
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=128)
