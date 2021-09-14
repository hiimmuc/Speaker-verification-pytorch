import glob
import os
import random
from argparse import Namespace

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from scipy import signal
from scipy.io import wavfile
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset

from utils import *


class Loader(Dataset):
    def __init__(self, dataset_file_name, max_frames):

        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames

        # Read Training Files...
        with open(dataset_file_name) as dataset_file:
            lines = dataset_file.readlines()

        dictkeys = list(set([x.split()[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.label_dict = {}
        self.data_list = []
        self.data_label = []

        for lidx, line in enumerate(lines):
            data = line.strip().split()

            speaker_label = dictkeys[data[0]]
            # filename = os.path.join(train_path, data[1])

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)

            self.data_label.append(speaker_label)
            # self.data_list.append(filename)
            self.data_list.append(data[1])

    def __getitem__(self, indices):
        feat = []

        for index in indices:
            if self.augment:
                augtype = random.randint(0, 4)
                if augtype == 0:
                    # non augment data
                    audio_name = self.data_list[index]
                else:
                    # randomly choose aomg 4 type of augment
                    audio_name = f"{self.data_list[index].replace('.wav', '')}_augmented_{augtype}.wav"
            else:
                audio_name = self.data_list[index]
            audio = loadWAV(audio_name,
                            self.max_frames,
                            evalmode=False)
            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size):
        self.data_source = data_source
        self.label_dict = data_source.label_dict
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size

    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()

        def lol(lst, sz): return [lst[i:i + sz]
                                  for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        # Data for each class
        for findex, key in enumerate(dictkeys):
            data = self.label_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk),
                                self.nPerSpeaker)

            rp = lol(
                np.random.permutation(len(data))[:numSeg], self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # Data in random order
        mixid = np.random.permutation(len(flattened_label))
        mixlabel = []
        mixmap = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = len(mixlabel) - len(mixlabel) % self.batch_size
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        return iter([flattened_list[i] for i in mixmap])

    def __len__(self):
        return len(self.data_source)


def get_data_loader(dataset_file_name, batch_size, max_frames, max_seg_per_spk, nDataLoaderThread,
                    nPerSpeaker, **kwargs):
    train_dataset = Loader(dataset_file_name, max_frames)

    train_sampler = Sampler(train_dataset, nPerSpeaker,
                            max_seg_per_spk, batch_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=nDataLoaderThread,
        sampler=train_sampler,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        drop_last=True,
    )

    return train_loader
