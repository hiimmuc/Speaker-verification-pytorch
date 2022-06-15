import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.utils import data


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


class FRM(nn.Module):
    def __init__(self, nb_dim, do_add=True, do_mul=True):
        super(FRM, self).__init__()
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        if self.do_mul:
            x = x * y
        if self.do_add:
            x = x + y
        return x


class AFMS(nn.Module):
    """
    Alpha-Feature map scaling, added to the output of each residual block[1,2].
    Reference:
    [1] RawNet2 : https://www.isca-speech.org/archive/Interspeech_2020/pdfs/1011.pdf
    [2] AMFS    : https://www.koreascience.or.kr/article/JAKO202029757857763.page
    """

    def __init__(self, nb_dim, do_add=True, do_mul=True):
        super(AFMS, self).__init__()
        self.alpha = nn.Parameter(torch.ones((nb_dim, 1)))
        self.fc = nn.Linear(nb_dim, nb_dim)
        self.sig = nn.Sigmoid()
        self.do_add = do_add
        self.do_mul = do_mul

    def forward(self, x):
        y = F.adaptive_avg_pool1d(x, 1).view(x.size(0), -1)
        y = self.sig(self.fc(y)).view(x.size(0), x.size(1), -1)

        x = x + self.alpha
        x = x * y
        # if self.do_mul:
        #     x = x * y
        # if self.do_add:
        #     x = x + y      
            
        return x


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
                nn.Conv1d(inplanes, planes, kernel_size=1,
                          stride=1, bias=False)
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
        low_hz = 10
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
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size) # hanning window

        # (1, kernel_size/2)
        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate  # Due to symmetry, I only need half of the time axes

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


class Residual_block_wFRM(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block_wFRM, self).__init__()
        self.first = first
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                             out_channels=nb_filts[1],
                                             padding=0,
                                             kernel_size=1,
                                             stride=1)

        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)
        self.frm = FRM(
            nb_dim=nb_filts[1],
            do_add=True,
            do_mul=True)

    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x

        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        out = self.frm(out)
        return out

## SAP
class Classic_Attention(nn.Module):
    def __init__(self,input_dim, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.lin_proj = nn.Linear(input_dim,embed_dim)
        self.v = torch.nn.Parameter(torch.randn(embed_dim))
    
    def forward(self,inputs):
        lin_out = self.lin_proj(inputs)
        v_view = self.v.unsqueeze(0).expand(lin_out.size(0), len(self.v)).unsqueeze(2)
        attention_weights = torch.tanh(lin_out.bmm(v_view).squeeze())
        attention_weights_normalized = F.softmax(attention_weights,1)
        return attention_weights_normalized


class ScaledDotProduct_attention(nn.Module):
    """
    Scaled dot product attention
    """

    def __init__(self, embed_dim, attn_dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_dropout = attn_dropout
        self.scaling = self.embed_dim ** -0.5
    
    
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        
        self.reset_parameters()
        
        self.in_proj_weight = Parameter(torch.Tensor(2 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(2 * embed_dim))
        
    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)
        
    def in_proj_qk(self, query):
        return self._in_proj(query).chunk(2, dim=-1)
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
        
    def forward(self,query,key):
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        q, k = self.in_proj_qk(query)
        q *= self.scaling
        return q,k
        attn_weights = torch.bmm(q, k.transpose(1, 2))    
        return attn_weights
    

class RawNet2(nn.Module):
    def __init__(self, filters, nb_classes=400,
                 nb_gru_layer=1, gru_node=1024,
                 nb_fc_node=512,
                 first_conv_size=251,
                 in_channels=1,
                 nb_samp=16240,
                 **kwargs):

        super(RawNet2, self).__init__()
        hoplength = 10e-3 * int(kwargs['sample_rate'])
        winlength = 25e-3 * int(kwargs['sample_rate'])
        
        nb_samp = int(int(kwargs['sample_rate']) * (kwargs['max_frames']/100)) + int(winlength - hoplength)
        # self.device = device
        self.ln = LayerNorm(nb_samp)
        self.first_conv = SincConv_fast(in_channels=in_channels,
                                        out_channels=filters[0],
                                        kernel_size=first_conv_size
                                        )

        self.first_bn = nn.BatchNorm1d(num_features=filters[0])

        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.block0 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[1], first=True))
        self.block1 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[1]))

        self.block2 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[2]))
        filters[2][0] = filters[2][1]
        self.block3 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[2]))
        self.block4 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[2]))
        self.block5 = nn.Sequential(Residual_block_wFRM(nb_filts=filters[2]))
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.bn_before_gru = nn.BatchNorm1d(num_features=filters[2][-1])

        self.gru = nn.GRU(input_size=filters[2][-1],
                          hidden_size=gru_node,
                          num_layers=nb_gru_layer,
                          batch_first=True)

        self.fc1_gru = nn.Linear(in_features=gru_node,
                                 out_features=nb_fc_node)
        self.fc2_gru = nn.Linear(in_features=nb_fc_node,
                                 out_features=nb_classes,
                                 bias=True)

        self.sig = nn.Sigmoid()

    def forward(self, x):
        # follow sincNet recipe
        #         x = x.unsqueeze(1)
        #         print(x.shape)
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = self.ln(x)
        x = x.view(nb_samp, 1, len_seq)
        x = F.max_pool1d(torch.abs(self.first_conv(x)), 3)
        x = self.first_bn(x)
        x = self.lrelu_keras(x)

        x = self.block0(x)
        x = self.block1(x)

        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.bn_before_gru(x)
        x = self.lrelu_keras(x)
        x = x.permute(0, 2, 1)  # (batch, filt, time) >> (batch, time, filt)
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = x[:, -1, :]
        code = self.fc1_gru(x)
        return code


def MainModel(nOut=512, **kwargs):

    nb_filters = [128, [128, 128], [128, 256], [256, 256]]

    model = RawNet2(filters=nb_filters,
                    nb_classes=nOut,
                    **kwargs
                    )
    return model


if __name__ == "__main__":
    from torchsummary import summary

    model = MainModel()
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))

    summary(model, (16240,), batch_size=128)
