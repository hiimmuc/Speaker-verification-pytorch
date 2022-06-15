from models.ResNetBaseline import ResNetSE
from models.ResNetBlocks import SEBasicBlockV2


def MainModel(nOut=256, **kwargs):
    # Number of filters
    num_filters = [32, 64, 128, 256]
    model = ResNetSE(SEBasicBlockV2, [3, 4, 6, 3], num_filters, nOut, **kwargs)
    return model


if __name__ == '__main__':

    pass
