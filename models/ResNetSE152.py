from models.ResNetBaseline import ResNetSE
from models.ResNetBlocks import SEBottleneck


def MainModel(nOut=512, **kwargs):
    # Number of filters
    num_filters = [64, 128, 256, 512]
    num_layers = [3, 8, 36, 3]
    model = ResNetSE(SEBottleneck, num_layers, num_filters, nOut, **kwargs)
    return model

if __name__ == '__main__':

    pass
