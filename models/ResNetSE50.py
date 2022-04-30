from models.ResNetBaseline import ResNetSE
from models.ResNetBlocks import SEBottleneck


def MainModel(nOut=512, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBottleneck, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model

if __name__ == '__main__':

    pass
