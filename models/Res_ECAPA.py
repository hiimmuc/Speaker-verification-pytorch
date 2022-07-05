import torch
import torch.nn as nn
import torch.nn.functional as F

from models.SpecAugment.specaugment import SpecAugment
from models.ECAPA_TDNN import *
from models.ResNetBlocks import SEBasicBlock, SEBottleneck


## baseline for se resnet
class ResNetSE_no_head(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_filters,
                 nOut,
                 encoder_type='ASP',
                 n_mels=80,
                 att_dim=128,
                 **kwargs):
        super(ResNetSE_no_head, self).__init__()

        print('Embedding size is %d, encoder %s.' % (nOut, encoder_type))
        self.aug = None if 'augment' in kwargs else kwargs['augment']
        self.aug_chain = None if 'augment_chain' in kwargs else kwargs['augment_chain']
        self.inplanes = num_filters[0]
        self.encoder_type = encoder_type
        self.n_mels = n_mels

        self.conv1 = nn.Conv2d(1,
                               num_filters[0],
                               kernel_size=3,
                               stride=(2,1),
                               padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        self.layers = []
        
        self.layers.append(self._make_layer(block, num_filters[0], layers[0]))
        for i in range(len(layers) - 1):
            self.layers.append(self._make_layer(block, num_filters[i + 1], 
                                                layers[i+1], 
                                                stride=(1, 1)))
        self.resnet_se_module = nn.Sequential(*self.layers)
        
        self.conv2 = nn.Conv2d(num_filters[-1],
                               nOut,
                               kernel_size=3,
                               stride=(2,1),
                               padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nOut)
                               
        self.specaug = SpecAugment()
        self.instancenorm = nn.InstanceNorm1d(n_mels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=stride,
                          bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        with torch.no_grad():
            x = x + 1e-6
            x = x.log()   
            x = x - torch.mean(x, dim=-1, keepdim=True)
            if self.aug and 'spec_domain' in self.aug_chain:
                x = self.specaug(x)
                
        x = self.instancenorm(x).unsqueeze(1)

        assert len(x.size()) == 4  # batch x Channels x n_mels x n_frames 
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.resnet_se_module(x)
                               
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.bn2(x)
                               
        return x

class ECAPA_TDNN_core(torch.nn.Module):
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
    speechbrain settings
    torch.Size([5, 1, 192])
    channels: [1024, 1024, 1024, 1024, 3072]
    kernel_sizes: [5, 3, 3, 3, 1]
    dilations: [1, 2, 3, 4, 1]
    groups: [1, 1, 1, 1, 1]
    attention_channels: 128
    lin_neurons: 192
    """

    def __init__(
        self,
        input_size=80,
        device="cpu",
        lin_neurons=192,
        activation=torch.nn.ReLU,
        channels=[512, 512, 512, 512, 1536],
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
        
        self.blocks = nn.ModuleList()

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

        # Attentive Statistical Pooling
        self.asp = AttentiveStatisticsPooling(
            channels[-1],
            attention_channels=attention_channels,
            global_context=global_context,
        )
        self.asp_bn = BatchNorm1d(input_size=channels[-1] * 2)

        # Final linear transformation
        self.fc = Conv1d(
            in_channels=channels[-1] * 2,
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
        # x shape: batch x n_mels x n_frames of batch x fea x time
        b, c, f, t = x.size()
        x = x.view((b, c * f, t))
       
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
        
        # Attentive Statistical Pooling
        x = self.asp(x, lengths=lengths)
        x = self.asp_bn(x)

        # Final linear transformation
        x = self.fc(x)
        x = x.squeeze()
       
        return x


def MainModel(nOut=128, **kwargs):
    num_filters = [32, 64]
    res_blocks = ResNetSE_no_head(SEBasicBlock, [1, 1], num_filters, num_filters[-1], **kwargs)
    
    n_features = kwargs['n_mels']
    input_size_tdnn = int(num_filters[-1] * n_features * (2 ** (-1 * len(num_filters))))
    
    ecapa_blocks = ECAPA_TDNN_core(input_size=input_size_tdnn, device="cpu",  
                                   lin_neurons=nOut, activation=torch.nn.ReLU,
                                   channels=[512, 512, 512, 512, 1536],        
                                   kernel_sizes=[5, 3, 3, 3, 1],
                                   dilations=[1, 2, 3, 4, 1],
                                   attention_channels=128,
                                   res2net_scale=8,
                                   se_channels=128,
                                   global_context=True)
    
    model = nn.Sequential(res_blocks,
                          ecapa_blocks)
    return model


# from torchsummary import  summary
# net = ECAPA_TDNN(23)
# print(net)
# summary(net.cuda(), (100, 23))