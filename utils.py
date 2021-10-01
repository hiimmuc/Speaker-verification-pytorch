import glob
import os
import random
from argparse import Namespace

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import yaml
from scipy import signal
from sklearn import metrics


def loadWAV(filename, max_frames, evalmode=True, num_eval=10, sr=None):
    '''Load audio form .wav file and return as the np arra

    Args:
        filename (str): [description]
        max_frames ([type]): [description]
        evalmode (bool, optional): [description]. Defaults to True.
        num_eval (int, optional): [description]. Defaults to 10.
        sr ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    '''
    # Maximum audio length
    # hoplength is 160, winlength is 400 -> total length  = winlength- hop_length + max_frames * hop_length
    max_audio = max_frames * 160 + 240

    audio, sample_rate = sf.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    if evalmode:
        # get 10 of audio and stack together
        startframe = np.linspace(0, audiosize - max_audio, num=num_eval)
    else:
        # get randomly initial index of frames, not always from 0
        startframe = np.array(
            [np.int64(random.random() * (audiosize - max_audio))])

    feats = []
    if evalmode and max_frames == 0:
        feats.append(audio)
    else:
        for asf in startframe:
            feats.append(audio[int(asf):int(asf) + max_audio])

    feat = np.stack(feats, axis=0).astype(np.float)
    if sr:
        return feat, sample_rate
    return feat


def round_down(num, divisor):
    return num - (num % divisor)


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class AugmentWAV(object):
    def __init__(self, musan_path, rir_path, max_frames):
        self.max_frames = max_frames
        self.max_audio = max_frames * 160 + 240

        self.noisetypes = ['noise', 'speech', 'music']

        self.noisesnr = {
            'noise': [0, 15],
            'speech': [13, 20],
            'music': [5, 15]
        }
        self.numnoise = {'noise': [1, 1], 'speech': [3, 7], 'music': [1, 1]}
        self.noiselist = {}

        augment_files = glob.glob(os.path.join(musan_path, '*/*/*/*.wav'))

        for file in augment_files:
            if not file.split('/')[-4] in self.noiselist:
                self.noiselist[file.split('/')[-4]] = []
            self.noiselist[file.split('/')[-4]].append(file)

        self.rir_files = glob.glob(os.path.join(
            rir_path, '*/*/*/*.wav'))

    def additive_noise(self, noisecat, audio):
        clean_db = 10 * np.log10(np.mean(audio ** 2) + 1e-4)

        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat],
                                  random.randint(numnoise[0], numnoise[1]))

        noises = []

        for noise in noiselist:
            noiseaudio = loadWAV(noise, self.max_frames, evalmode=False)
            noise_snr = random.uniform(self.noisesnr[noisecat][0],
                                       self.noisesnr[noisecat][1])
            noise_db = 10 * np.log10(np.mean(noiseaudio[0] ** 2) + 1e-4)
            noises.append(
                np.sqrt(10 ** ((clean_db - noise_db - noise_snr) / 10)) *
                noiseaudio)
        aug_audio = np.sum(np.concatenate(noises, axis=0),
                           axis=0, keepdims=True) + audio
        return aug_audio

    def reverberate(self, audio):
        rir_file = random.choice(self.rir_files)

        rir, _ = sf.read(rir_file)
        rir = np.expand_dims(rir.astype(np.float), 0)
        rir = rir / np.sqrt(np.sum(rir ** 2))
        aug_audio = signal.convolve(audio, rir, mode='full')[
            :, :self.max_audio]
        return aug_audio


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # tensor.topk -> tensor, long tensor, return the k largest values along dim
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # Computes element-wise equality
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # calculate number of true values/ batchsize
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size(
        )) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        return F.conv1d(input, self.flipped_filter).squeeze(1)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    idxG = np.argmax(gmean)
    print(f"G-mean at {idxG}: {gmean[idxG]}, Best Threshold {thresholds[idxG]}")

    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))  # index of min fpr - fnr = fpr + tpr - 1
    eer = np.mean([fpr[idxE], fnr[idxE]])  # EER in % = (fpr + fnr) /2
    optimal_threshold = thresholds[idxE]

    return (tunedThreshold, eer, optimal_threshold, metrics.auc(fpr, tpr))


def score_normalization(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """

    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)


def cosine_simialrity(ref, com):
    return np.mean(abs(F.cosine_similarity(ref, com, dim=1)).cpu().numpy())


def read_config(config_path, args=None):
    if args is None:
        args = Namespace()
    with open(config_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        args.__dict__[k] = v
    return args


if __name__ == '__main__':
    pass
