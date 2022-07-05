from models.ResNetBaseline import ResNetSE
from models.ResNetBlocks import SEBottleneck


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [128, 128, 256, 256]
    num_layers = [6, 16, 24, 3]
    model = ResNetSE(SEBottleneck, num_layers, num_filters, nOut, **kwargs)
    return model

if __name__ == '__main__':

    pass
