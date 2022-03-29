"""
Multi tasks learning with multi proxies anchor balance loss and am softmax
"""

import torch
import torch.nn as nn

import losses.MMP_Balance as mmp_balance
import losses.AmSoftmax as amsoftmax


class MMP_Balance_MTL(nn.Module):

    def __init__(self, **kwargs):
        super(MMP_Balance_MTL, self).__init__()

        self.test_normalize = True

        self.amsoftmax = amsoftmax.AmSoftmax(**kwargs)
        self.mmp_balance = mmp_balance.MMP_Balance(**kwargs)

        print('Initialised Softmax Multi Mask Proxy Loss')

    def forward(self, x, label=None):

        weight = 0.6
        nlossCE, prec1 = self.amsoftmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(x.size()[1]))

        nlossML, prec2 = self.mmp_balance(x, label)
        
        return (1 - weight) * nlossCE + weight * nlossML, prec1