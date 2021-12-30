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
from models import RawNet2, RawNet2v2

# torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True


class RawNet2v5(nn.Module):
    """
    Refactored RawNet2 architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    """

    def __init__(self,nOut=512, **kwargs):
        super(RawNet2v5, self).__init__()
        self.rawnet2_origin = RawNet2.MainModel(nOut=nOut,**kwargs)
        self.rawnet2v2 = RawNet2v2.MainModel(nOut=nOut,**kwargs)


    def forward(self, x):
        #####
        
        # #####
        # # forward model 1
        # #####
        out1 = self.rawnet2_origin(x)
        
        # #####
        # # forward model 2
        # #####
        out2 = self.rawnet2v2(x)
        #
        out = torch.cat([out1.unsqueeze(-1), out2.unsqueeze(-1)], dim=-1)
        out = torch.mean(out, dim=-1)

        return out


def MainModel(nOut=512, **kwargs):
    model = RawNet2v5(nOut=nOut,**kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
