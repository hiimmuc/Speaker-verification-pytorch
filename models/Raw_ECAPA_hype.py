import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

from models import ECAPA_TDNN, RawNet2_custom


class Raw_ECAPA(nn.Module):
    """
    Refactored RawNet2 combined with ECAPA architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143)
    """

    def __init__(self, nOut=512, **kwargs):
        super(Raw_ECAPA, self).__init__()
        self.ECAPA_TDNN = ECAPA_TDNN.MainModel(nOut=192, channels= [512, 512, 512, 512, 1536], **kwargs)
        self.rawnet2v2 = RawNet2_custom.MainModel(nOut=nOut-192,
                                                  front_proc='sinc',  aggregate='gru',
                                                  att_dim=128, **kwargs)
        
        features = 'melspectrogram'
        Features_extractor = importlib.import_module(
            'models.FeatureExtraction.feature').__getattribute__(f"{features}")
        self.compute_features = Features_extractor(**kwargs) 
        
        att_size = 128
        self.bn_before_agg = nn.BatchNorm1d(nOut)
        self.attention = nn.Sequential(
            nn.Conv1d(nOut, att_size, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(att_size),
            nn.Conv1d(att_size, nOut, kernel_size=1),
            nn.Softmax(dim=-1),
        )
        self.fc = nn.Linear(nOut * 2, 512)
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        #####
        
        # #####
        # # forward model 1
        # #####

        x_spec = self.compute_features(x)
        out1 = self.ECAPA_TDNN(x_spec)
        
        # #####
        # # forward model 2
        # #####
        out2 = self.rawnet2v2(x)
        #
        out = torch.cat([out1, out2], dim=-1).unsqueeze(-1) # bs, nOut -> bs, nOut, 1
        
        out = self.bn_before_agg(out)
        
        out = self.lrelu(out)
        
        w = self.attention(out)
        
        m = torch.sum(out * w, dim=-1)
        s = torch.sqrt(
            (torch.sum((out ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        out = torch.cat([m, s], dim=1)
        out = out.view(out.size(0), -1)       

        #####
        # speaker embedding layer
        #####
        out = self.fc(out)
        out = out.squeeze()
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
