from traitlets import default
from utils import read_config
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib

from models import *


class Mixed_model(nn.Module):
    """
    Concat types of models without reimplements
    """

    def __init__(self, **kwargs):
        super(Mixed_model, self).__init__()
        self.model_options = kwargs['model_options']

        self.modules = []
        for model_name, feature_type, nOut in zip(self.model_options['name'], self.model_options['feature_type'],
                                                  self.model_options['nOut']):

            if feature_type in ['mfcc', 'melspectrogram']:
                Features_extractor = importlib.import_module(
                    'models.FeatureExtraction.feature').__getattribute__(f"{feature_type.lower()}")
                compute_features = Features_extractor(**kwargs)
            else:
                compute_features = None

            SpeakerNetModel = importlib.import_module(
                'models.' + model_name).__getattribute__('MainModel')
            __S__ = SpeakerNetModel(
                nOut=nOut, features=feature_type, **kwargs)

            if compute_features is not None:
                self.modules.append(nn.Sequential(
                    compute_features, __S__))
            else:
                self.modules.append(__S__)

        att_size = 128
        temp_emb_size = sum(self.model_options['nOut'])
        self.bn_before_agg = nn.BatchNorm1d(temp_emb_size)
        self.attention = nn.Sequential(
            nn.Conv1d(temp_emb_size, att_size, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(att_size),
            nn.Conv1d(att_size, temp_emb_size, kernel_size=1),
            nn.Softmax(dim=-1),
        )
        self.fc = nn.Linear(temp_emb_size * 2, 512)
        self.lrelu = nn.LeakyReLU(0.3)

    def forward(self, x):
        #####
        out_feats = []
        for module in self.modules:
            out_feats.append(module(x))

        out = torch.cat(out_feats, dim=-1)

        #####
        # aggregation: attentive statistical pooling
        #####
        out = self.bn_before_agg(out)
        out = self.lrelu(out)
        w = self.attention(out)
        m = torch.sum(out * w, dim=-1)
        s = torch.sqrt(
            (torch.sum((out ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        out = torch.cat([m, s], dim=1)
        out = x.view(out.size(0), -1)

        #####
        # speaker embedding layer
        #####
        out = self.fc(out)
        out = out.squeeze()

        return out


def MainModel(**kwargs):
    model = Mixed_model(**kwargs)
    return model


parser = argparse.ArgumentParser(description="SpeakerNet")

if __name__ == "__main__":
    from torchsummary import summary

    parser.add_argument('--config', type=str, default=None)
    sys_args = parser.parse_args()

    if sys_args.config is not None:
        args = read_config(sys_args.config, sys_args)
        args = argparse.Namespace(**args)
    model_options = args.model
    model = MainModel(model_options=model_options, **vars(args))
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=2)
