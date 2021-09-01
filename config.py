class mfcc_config:
    sampling_rate = 16000
    max_pad_length = 400  # calculate by avg time length / time overlap - 1 for st :v
    n_mfcc = 400  # win_length
    n_fft = 512
    hop_length = 160
    n_mels = 64
    max_samples = 100  # 300 ~ 3 seconds

# TODO: add training configuration for hyper params


class training_config:
    batchsize = 128
    learning_rate = 0.001
    optimizer = 'Adam'


NUM_CLASSES = 400
FEATURE_DIM = (mfcc_config.n_mels, mfcc_config.max_pad_length)


class MultiSimilarityLoss:
    scale_pos = 2
    scale_neg = 1
