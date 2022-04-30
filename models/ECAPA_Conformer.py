import torch  # noqa: F401
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.ECAPA_TDNN import *
from models.ECAPA_utils import Conv1d as _Conv1d
from models.ECAPA_utils import BatchNorm1d as _BatchNorm1d
from models.conformer.conformer.encoder import ConformerEncoder


class ECAPA_Conformer(torch.nn.Module):
    """An implementation of the speaker embedding model in a paper.
    "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in
    TDNN Based Speaker Verification" (https://arxiv.org/abs/2005.07143).
    Arguments
    ---------
    device : str
        Device used, e.g., "cpu" or "cuda".
    activation : torch class
        A class for constructing the activation layers.
    channels : list of ints
        Output channels for TDNN/SERes2Net layer.
    kernel_sizes : list of ints
        List of kernel sizes for each layer.
    dilations : list of ints
        List of dilations for kernels in each layer.
    lin_neurons : int
        Number of neurons in linear layers.
    Example
    -------
    >>> input_feats = torch.rand([5, 120, 80])
    >>> compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    >>> outputs = compute_embedding(input_feats)
    >>> outputs.shape
    torch.Size([5, 1, 192])
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.GELU,
        channels=[1024, 1024, 1024, 1024, 3072],
        kernel_sizes=[5, 3, 3, 3, 1],
        dilations=[1, 2, 3, 4, 1],
        attention_channels=128,
        res2net_scale=8,
        se_channels=128,
        global_context=True,
        **kwargs
    ):

        super().__init__()
        assert len(channels) == len(kernel_sizes)
        assert len(channels) == len(dilations)
        self.channels = channels
        self.aug = kwargs['augment']
        self.aug_chain = kwargs['augment_chain']
        sample_rate = int(kwargs['sample_rate'])
        hoplength = int(10e-3 * sample_rate)
        winlength = int(25e-3 * sample_rate)
        n_frames = sample_rate//hoplength + 2
        n_mels = kwargs['n_mels']
        input_size = n_mels
        
        self.blocks = nn.ModuleList()
        
        self.specaug = SpecAugment() # Spec augmentation

        # The initial TDNN layer
        self.blocks.append(
            TDNNBlock(
                input_size,
                channels[0],
                kernel_sizes[0],
                dilations[0],
                activation,
            )
        )

        # SE-Res2Net layers
        for i in range(1, len(channels) - 1):
            self.blocks.append(
                SERes2NetBlock(
                    channels[i - 1],
                    channels[i],
                    res2net_scale=res2net_scale,
                    se_channels=se_channels,
                    kernel_size=kernel_sizes[i],
                    dilation=dilations[i],
                    activation=activation,
                )
            )

        # Multi-layer feature aggregation
        self.mfa = TDNNBlock(
            channels[-1],
            channels[-1],
            kernel_sizes[-1],
            dilations[-1],
            activation,
        )
        
        # Conformer
        self.conformer_block = ConformerEncoder(
            input_dim = channels[-1],
            encoder_dim = lin_neurons,
            num_layers = 16,
            num_attention_heads = 8,
            feed_forward_expansion_factor = 4,
            conv_expansion_factor = 2,
            input_dropout_p = 0.1,
            feed_forward_dropout_p = 0.1,
            attention_dropout_p = 0.1,
            conv_dropout_p = 0.1,
            conv_kernel_size = 3,
            half_step_residual = True
        )
        
        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            lin_neurons,
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=(lin_neurons * 2))

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=lin_neurons * 2,
            out_channels=lin_neurons,
            kernel_size=1,
        )

    def forward(self, x, lengths=None):
        """Returns the embedding vector.
        Arguments
        ---------
        x : torch.Tensor
            Tensor of shape (batch, time, channel).
        """
        # Minimize transpose for efficiency
        
        with torch.no_grad():
            x = x + 1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if self.aug and 'spec_domain' in self.aug_chain:
                x = self.specaug(x)
        
        # x shape: batch x n_mels x n_frames of batch x fea x time

        xl = []
        for layer in self.blocks:
            try:
                x = layer(x, lengths=lengths)
            except TypeError:
                x = layer(x)
            xl.append(x)

        # Multi-layer feature aggregation
        x = torch.cat(xl[1:], dim=1)
        x = self.mfa(x)

        # conformer
        x = x.transpose(1, -1)
        x = self.conformer_block(x)
        x = x.transpose(1, -1)

        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)
        
        # Final linear transformation
        x = self.fc(x)
        x = x.squeeze()
       
        return x
    
def MainModel(nOut=192, **kwargs):
    model = ECAPA_Conformer(lin_neurons=nOut, **kwargs)
    return model