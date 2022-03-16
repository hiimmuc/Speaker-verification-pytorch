import math
from collections import OrderedDict

import numpy as np
import torch
import torch.backends.cudnn
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from numpy.core.fromnumeric import transpose
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data
from models import ECAPA_TDNN, RawNet2v2

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Raw_ECAPA(nn.Module):
    """
    Refactored RawNet2 combined with ECAPA architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143)
    """

    def __init__(self,nOut=512, **kwargs):
        super(Raw_ECAPA, self).__init__()
        self.ECAPA_TDNN = ECAPA_TDNN.MainModel(nOut=192,**kwargs)
        self.rawnet2v2 = RawNet2v2.MainModel(nOut=nOut-192,**kwargs)

    def forward(self, x):
        #####
        
        # #####
        # # forward model 1
        # #####
        out1 = self.ECAPA_TDNN(x)
        
        # #####
        # # forward model 2
        # #####
        out2 = self.rawnet2v2(x)
        #

#         out = torch.cat([out1.squeeze(), out2.squeeze()], dim=-1).unsqueeze(0)
        out = torch.cat([out1, out2], dim=-1)
#         out = torch.mean(out, dim=-1)
        return out


def MainModel(nOut=512, **kwargs):
    model = Raw_ECAPA(nOut=nOut,**kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
