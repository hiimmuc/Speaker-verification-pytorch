# import torch
# import torch.nn.functional as F
# import time
# x = torch.randn((256, 96, 128, 128)).cuda()
# t = time.time()
# x = F.avg_pool2d(x, x.size()[2:])

# %timeit F.adaptive_avg_pool2d(x, (1, 1))

# %timeit torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)
from utils import *

path = './dataset/wavs/272-M-26/speaker_272-10.wav'
audio = loadWAV(path, 100, evalmode=False)
print(audio.shape)
