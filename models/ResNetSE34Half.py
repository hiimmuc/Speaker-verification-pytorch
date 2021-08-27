from torchsummary import summary

from models.ResNetBlocks import *
from models.ResNetSE34L import *


def MainModel(nOut=256, summary_model=True, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    if summary_model:
        summary(model, (3, 64, 400), 128)
    return model
