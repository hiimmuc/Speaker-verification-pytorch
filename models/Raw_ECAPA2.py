import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
from models import ECAPA_TDNN, RawNet2v3

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


class Raw_ECAPA(nn.Module):
    """
    Combination of RawNet2 and ECAPA-TDNN architecture.
    """

    def __init__(self,nOut=896, test_recog=False, **kwargs):
        super(Raw_ECAPA, self).__init__()
        self.ECAPA_TDNN = ECAPA_TDNN.MainModel(nOut=192,**kwargs)
        self.rawnet2v3 = RawNet2v3.MainModel(nOut=nOut-192,**kwargs)
        self.test_recog = test_recog
        self.nClasses = kwargs['nClasses']
        self.fc_recog = nn.Linear(in_features = nOut, out_features = self.nClasses, bias = True)

    def forward(self, x):
        #####
        
        # #####
        # # forward model 1
        # #####
        out1 = self.ECAPA_TDNN(x)
        
        # #####
        # # forward model 2
        # #####
        out2 = self.rawnet2v3(x)
        #
        out = torch.cat([out1, out2], dim=-1)
#         out = torch.mean(out, dim=-1)
        if self.test_recog:
            out = self.fc_recog(out)
            
        return out


def MainModel(nOut=896, **kwargs):
    model = Raw_ECAPA(nOut=nOut,**kwargs)
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
