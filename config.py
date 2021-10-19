class DataLoaderCfg:
    max_frames = 100
    eval_frame = 100
    batch_size = 128
    max_seg_per_spk = 100
    nDataLoaderThread = 2


class TrainingHyperParams:
    device = 'cpu'
    test_interval = 10
    model = 'ResNetSE34V2'
    max_epoch = 500
    criterion = 'amsoftmax'


class ModelParams:
    save_path = 'exp'
    model = 'ResNetSE34V2'
    optimizer = 'adam'
    callbacks = 'steplr'
    criterion = 'amsoftmax'
    device = 'cpu'
    max_epoch = 500
    nOut = 512
    nClasses = 400


# pr = ModelParams()
# print(vars(pr))
