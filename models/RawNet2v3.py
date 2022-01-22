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

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class RawNetBasicBlock(nn.Module):
    """
    Basic block for RawNet architectures.
    This block follows pre-activation[1].
    Arguments:
    downsample  : perform reduction in the sequential(time) domain
                  (different with shortcut)
    Reference:
    [1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks. arXiv:1603.05027
    """

    def __init__(self, inplanes, planes, downsample=None):
        super(RawNetBasicBlock, self).__init__()
        self.downsample = downsample

        #####
        # core layers
        #####
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.mp = nn.MaxPool1d(3)
        self.afms = AFMS(planes)
        self.lrelu = nn.LeakyReLU(0.3)

        #####
        # settings
        #####
        if inplanes != planes:  # if change in number of filters
            self.shortcut = nn.Sequential(
                nn.Conv1d(inplanes, planes, kernel_size=1, stride=1, bias=False)
            )

    def forward(self, x):
        out = self.lrelu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, "shortcut") else x
        out = self.conv1(out)
        out = self.conv2(self.lrelu(self.bn2(out)))
        out = out + shortcut

        if self.downsample:
            out = self.mp(out)
        out = self.afms(out)

        return out


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim):
        super(AFMS, self).__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        return x


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast, self).__init__()

        if in_channels != 1:
            # msg = (f'SincConv only support one input channel '
            #       f'(here, in_channels = {in_channels:d}).')
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        # filter lower frequency (out_channels, 1)
        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))

        # filter frequency band (out_channels, 1)
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin = torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))  # computing only half of the window
        self.window_ = 0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size) # hanning window

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low_hz_ = self.low_hz_.to(waveforms.device)
        band_hz_ = self.band_hz_.to(waveforms.device)

        low = self.min_low_hz + torch.abs(low_hz_)

        high = torch.clamp(low + self.min_band_hz + torch.abs(band_hz_), self.min_low_hz, self.sample_rate/2)
        band = (high-low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations.
        band_pass_left = ((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_
        band_pass_center = 2*band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)

        band_pass = band_pass / (2*band[:, None])

        self.filters_map = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters_map, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class RawNet2(nn.Module):
    """
    Refactored RawNet2 architecture.
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    """

    def __init__(
            self,
            block,
            layers,
            nb_filters,
            code_dim=512,
            in_channels=1,
            n_mels=64,
            log_input=True,
            nb_gru_layers=1, 
            gru_node=1024,
            first_conv_size=251,
            nb_samp=16240,
            **kwargs):
        super(RawNet2, self).__init__()
#         self.device = kwargs['device']
        self.inplanes = nb_filters[0]
        self.n_mels = n_mels
        self.log_input = log_input
        #####
        # first layers before residual blocks
        #####
        self.conv1 = nn.Conv1d(
            in_channels,
            nb_filters[0],
            kernel_size=3,
            stride=3,
            padding=0
        )
        #  sinc layer
        hoplength = 10e-3 * int(kwargs['sample_rate'])
        winlength = 25e-3 * int(kwargs['sample_rate'])
        
        nb_samp = int(int(kwargs['sample_rate']) * (kwargs['max_frames']/100)) + int(winlength - hoplength)
        self.ln = LayerNorm(nb_samp)
        self.first_conv = SincConv_fast(in_channels=in_channels,
                                        out_channels=nb_filters[0],
                                        kernel_size=first_conv_size,
                                        sample_rate=int(kwargs['sample_rate']),
                                        stride=1, 
                                        padding=0, 
                                        dilation=1, 
                                        bias=False, 
                                        groups=1, 
                                        min_low_hz=50, min_band_hz=50
                                        )

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
        # aggregate to utterance(segment)-level
        #####
        self.bn_before_agg = nn.BatchNorm1d(nb_filters[5])
        self.attention = nn.Sequential(
            nn.Conv1d(nb_filters[5], 128, kernel_size=1),
            nn.LeakyReLU(),
            nn.BatchNorm1d(128),
            nn.Conv1d(128, nb_filters[5], kernel_size=1),
            nn.Softmax(dim=-1),
        )

        #####
        # speaker embedding layer
        #####
        self.fc = nn.Linear(nb_filters[5] * 2, code_dim)  # speaker embedding layer

        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

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

        ##########
        self.bn_before_gru = nn.BatchNorm1d(nb_filters[5])

        self.gru = nn.GRU(input_size=nb_filters[5],
                          hidden_size=gru_node,
                          num_layers=nb_gru_layers,
                          batch_first=True)

        self.fc_after_gru = nn.Linear(in_features=gru_node,
                                      out_features=code_dim)
        ###########

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

    def forward(self, x):
        #####
        x = self.ln(x)
        x = x.unsqueeze(1)
        #####
        # first layers before residual blocks
        #####
#         x = self.conv1(x)
#
#         nb_samp = x.shape[0]
#         len_seq = x.shape[1]
#         x = self.ln(x)
#         x=x.view(nb_samp,1,len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)

        #####
        # frame-level
        #####
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)

        #####
        # gru
        #####

        out1 = self.bn_before_gru(x)
        out1 = self.lrelu_keras(out1)
        out1 = out1.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        out1, _ = self.gru(out1)
        out1 = out1[:, -1, :]
        out1 = self.fc_after_gru(out1)

        #####
        # aggregation: attentive statistical pooling
        #####
        out2 = self.bn_before_agg(x)
        out2 = self.lrelu_keras(out2)
        w = self.attention(out2)
        m = torch.sum(out2 * w, dim=-1)
        s = torch.sqrt((torch.sum((out2 ** 2) * w, dim=-1) - m ** 2).clamp(min=1e-5))
        out2 = torch.cat([m, s], dim=1)
        out2 = out2.view(out2.size(0), -1)
        # #####
        # # speaker embedding layer
        # #####
        out2 = self.fc(out2)
        
        #
        out = torch.cat([out1.unsqueeze(-1), out2.unsqueeze(-1)], dim=-1)
        out = torch.mean(out, dim=-1)

        return out


def MainModel(nOut=512, **kwargs):
    layers = [1, 1, 3, 4, 6, 3]
    nb_filters = [128, 128, 256, 256, 256, 256]

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
