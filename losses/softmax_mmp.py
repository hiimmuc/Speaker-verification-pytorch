import torch
import torch.nn as nn

import losses.mmp_balance as mmp_balance
import losses.softmax as softmax


class LossFunction(nn.Module):

    def __init__(self, **kwargs):
        super(LossFunction, self).__init__()

        self.test_normalize = True

        self.softmax = softmax.LossFunction(**kwargs)
        self.mmp_balance = mmp_balance.MMP_Balance2(**kwargs)

        print('Initialised Softmax Multi Mask Proxy Loss')

    def forward(self, x, label=None):

        assert x.size()[1] == 2

        nlossS, prec1 = self.softmax(
            x.reshape(-1, x.size()[-1]), label.repeat_interleave(2))

        nlossP, _ = self.mmp_balance(x, None)

        return nlossS + nlossP, prec1