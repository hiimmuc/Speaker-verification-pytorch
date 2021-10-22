import collections
import contextlib
import glob
import os
import random
import sys
import time
import wave
from argparse import Namespace

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import webrtcvad
import yaml
from matplotlib import pyplot as plt
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

    return feat if not sr else (feat, sample_rate)


def mels_spec_preprocess(feat):
    n_mels = 64
    instancenorm = nn.InstanceNorm1d(n_mels)

    torchfb = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=512,
            win_length=400,
            hop_length=160,
            window_fn=torch.hamming_window,
            n_mels=n_mels))

    feat = torch.FloatTensor(feat)

    with torch.no_grad():
        feat = torchfb(feat) + 1e-6
        feat = instancenorm(feat).unsqueeze(1)

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
            rir_path, '*/*/*.wav'))

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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# VAD utilities


def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class VAD:
    def __init__(self,  mode=3, frame_duration=30, win_length=300) -> None:
        self.mode = mode
        self.frame_duration = frame_duration
        self.win_length = win_length
        self.vad = webrtcvad.Vad(int(mode))

    def frame_generator(self, audio, sample_rate):
        """Generates audio frames from PCM audio data.
        Takes the desired frame duration in milliseconds, the PCM data, and
        the sample rate.
        Yields Frames of the requested duration.
        """
        frame_duration_ms = self.frame_duration
        n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / 2.0
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def vad_collector(self, sample_rate, frames, show=False):
        """Filters out non-voiced audio frames.
        Given a webrtcvad.Vad and a source of audio frames, yields only
        the voiced audio.
        Uses a padded, sliding window algorithm over the audio frames.
        When more than 90% of the frames in the window are voiced (as
        reported by the VAD), the collector triggers and begins yielding
        audio frames. Then the collector waits until 90% of the frames in
        the window are unvoiced to detrigger.
        The window is padded at the front and back to provide a small
        amount of silence or the beginnings/endings of speech around the
        voiced frames.
        Arguments:
        sample_rate - The audio sample rate, in Hz.
        frame_duration_ms - The frame duration in milliseconds.
        padding_duration_ms - The amount to pad the window, in milliseconds.
        vad - An instance of webrtcvad.Vad.
        frames - a source of audio frames (sequence or generator).
        Returns: A generator that yields PCM audio data.
        """
        padding_duration_ms = self.win_length
        frame_duration_ms = self.frame_duration
        num_padding_frames = int(padding_duration_ms / frame_duration_ms)
        # We use a deque for our sliding window/ring buffer.
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
        # NOTTRIGGERED state.
        triggered = False

        voiced_frames = []
        for frame in frames:
            is_speech = self.vad.is_speech(frame.bytes, sample_rate)
            if show:
                sys.stdout.write('1' if is_speech else '_')
            if not triggered:
                ring_buffer.append((frame, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                # If we're NOTTRIGGERED and more than 90% of the frames in
                # the ring buffer are voiced frames, then enter the
                # TRIGGERED state.
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    if show:
                        sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                    # We want to yield all the audio we see from now until
                    # we are NOTTRIGGERED, but we have to start with the
                    # audio that's already in the ring buffer.
                    for f, s in ring_buffer:
                        voiced_frames.append(f)
                    ring_buffer.clear()
            else:
                # We're in the TRIGGERED state, so collect the audio data
                # and add it to the ring buffer.
                voiced_frames.append(frame)
                ring_buffer.append((frame, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                # If more than 90% of the frames in the ring buffer are
                # unvoiced, then enter NOTTRIGGERED and yield whatever
                # audio we've collected.
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    if show:
                        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))

                    triggered = False
                    yield b''.join([f.bytes for f in voiced_frames])
                    ring_buffer.clear()
                    voiced_frames = []
        if triggered:
            if show:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
        if show:
            sys.stdout.write('\n')
        # If we have any leftover voiced audio when we run out of input,
        # yield it.
        if voiced_frames:
            yield b''.join([f.bytes for f in voiced_frames])

    def detect(self, audio_path, write=True, show=False):
        if not os.path.exists(audio_path):
            raise "Path is not existed"
        audio, sample_rate = read_wave(audio_path)

        frames = self.frame_generator(audio, sample_rate)
        frames = list(frames)

        segments = self.vad_collector(sample_rate, frames, show=show)

        if write:
            for i, segment in enumerate(segments):
                path = f"{audio_path.replace('.wav', '')}_vad_{i}.wav"
                write_wave(path, segment, sample_rate)

        segments = [np.frombuffer(seg) for seg in segments]
        return segments


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot loss graph along with training process

def plot_graph(data, x_label, y_label, title, save_path, show=True, color='b-', mono=True, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    if mono:
        plt.plot(data, color=color)
    else:
        for dt in data:
            plt.plot(dt)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()


def plot_from_file(model, show=False):
    '''Plot graph from score file

    Args:
        model (str): model name
        show (bool, optional): Whether to show the graph. Defaults to False.
    '''
    model_name = model
    with open(f"exp/{model_name}/result/scores.txt") as f:
        line_data = f.readlines()

    line_data = [line.strip().replace('\n', '').split(',')
                 for line in line_data]
    data = [{}]
    last_epoch = 1
    step = 10
    for line in line_data:
        if 'IT' in line[0]:
            epoch = int(line[0].split(' ')[-1])

            if epoch not in range(last_epoch - step, last_epoch + 2):
                data.append({})

            data[-1][epoch] = line
            last_epoch = epoch
    # print(data)
    for i, dt in enumerate(data):
        data_loss = [float(line[3].strip().split(' ')[1])
                     for _, line in dt.items()]
        plot_graph(data_loss, 'epoch', 'loss', 'Loss',
                   f"exp/{model_name}/result/loss_{i}.png", color='b', mono=True, show=show)
        data_acc = [float(line[2].strip().split(' ')[1])
                    for _, line in dt.items()]
        plot_graph(data_acc, 'epoch', 'accuracy', 'Accuracy',
                   f"exp/{model_name}/result/acc_{i}.png", color='r', show=show)
        plt.close()


if __name__ == '__main__':
    path = r'dataset\wavs\504-F-25\504-F-25.wav'
    t = time.time()
    # vad_engine = VAD()
    segments = VAD(win_length=100).detect(path, write=False, show=False)
    # print(sum([len(seg) for seg in segments]))
    audio = np.concatenate(segments)
    print(audio.shape, time.time() - t)

    t0 = time.time()
    audio2 = sf.read(path)[0]
    print(audio2.shape, time.time() - t0)
