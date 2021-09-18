
import argparse
import csv
import glob
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from tqdm.auto import tqdm

from models.RepVGG import *
from utils import *


# from models.ResNetSE34V2 import *
def calcualte_similarity_of_test_result():
    test = pd.read_csv(
        "exp/dump/submission_list_test11091050.csv")  # 86.55%

    ref = pd.read_csv(
        "exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")  # 87.38%
    # similarity accepted: 74% -> 100%
    data1 = list(ref['label'])
    data2 = list(test['label'])

    similarity = [1 if data1[i] == data2[i] else 0 for i in range(len(data1))]
    print(similarity.count(1))
    print(f"similarity: {similarity.count(1)/len(data1)}")


# model = MainModel(model='RepVGG-B0')

# test for new dataloader


class SpeakerDataset(Dataset):
    def __init__(self, args, max_frames=100, transform=None, target_transform=None) -> None:
        super(SpeakerDataset).__init__()
        self.args = args
        self.max_frames = max_frames

        self.transform = transform
        self.target_transform = target_transform

        self.feature_vector_path = os.path.join(
            self.args.data_dir, "feature_vectors")
        self.list_folders = os.listdir(self.feature_vector_path)

        if len(self.list_folders) == 0:
            raise ValueError(
                "No feature vectors found, please prepare dataset first!")

        self.list_folders.sort()
        # take the last update ones
        data_path = os.path.join(
            self.feature_vector_path, self.list_folders[-1])

        self.X = np.load(os.path.join(data_path, "data.npy"))
        self.y_one_hot = np.load(os.path.join(data_path, "label_onehot.npy"))
        self.y_int_encoded = np.load(
            os.path.join(data_path, "label_int_encode.npy"))
        self.y_raw = np.load(os.path.join(data_path, "label_raw.npy"))
        print('Done loading dataset from npy files')
        print('Data shape:', self.X.shape)
        print('Label onehot shape:', self.y_one_hot.shape)
        print('Label_int_encoded shape:', self.y_int_encoded.shape)
        print('Label_raw shape:', self.y_raw.shape)

        # save to dict for Datasampler

        dictkeys = list(set([x for x in self.y_raw.reshape(-1)]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.label_dict = {}
        self.data_list = []
        self.data_label = []
        # line: label data_path
        for lidx, label in enumerate(self.y_raw.reshape(-1)):

            speaker_label = dictkeys[label]

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)

            self.data_label.append(speaker_label)

            self.data_list.append(self.X[lidx])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.transform:
            self.X[idx] = self.transform(self.X[idx])
        if self.target_transform:
            self.y_one_hot[idx] = self.target_transform(
                self.y_one_hot[idx])
        # if self.data_list
        return self.X[idx], self.y_int_encoded[idx]


class SpeakerSampler(torch.utils.data.Sampler):
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


def get_data_loader(args, shuffle=True, **kwargs):
    batch_size = args.batch_size
    n_per_spk = args.nPerSpeaker
    n_dataloader_threads = args.nDataLoaderThread
    max_seg_per_spk = args.max_seg_per_spk

    dataset = SpeakerDataset(args,
                             max_frames=args.max_frames)

    sampler = SpeakerSampler(dataset,
                             n_per_spk,
                             max_seg_per_spk,
                             batch_size)

    # ValueError: sampler option is mutually exclusive with shuffle
    sampler = sampler if not shuffle else None

    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        sampler=sampler,
                        num_workers=n_dataloader_threads,
                        pin_memory=False,
                        worker_init_fn=worker_init_fn,
                        drop_last=True)
    return loader


# Test Data Loader
parser = argparse.ArgumentParser(description="Data loader")
if __name__ == '__main__':
    # Test for data loader
    parser.add_argument('--augment',
                        type=bool,
                        default=True,
                        help='decide whether use augment data')
    parser.add_argument('--data_dir',
                        type=str,
                        default="dataset/",
                        help='Directory to save files(parent root)')
    parser.add_argument('--train_list',
                        type=str,
                        default="dataset/train.txt",
                        help='Directory to save files(parent root)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=128,
                        help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk',
                        type=int,
                        default=100,
                        help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread',
                        type=int,
                        default=2,
                        help='Number of loader threads')
    parser.add_argument('--nPerSpeaker',
                        type=int,
                        default=2,
                        help='Number of utterances per speaker per batch, only for metric learning based losses'
                        )
    parser.add_argument('--max_frames',
                        type=int,
                        default=100,
                        help='Input length to the network for training')
    args = parser.parse_args()
    t = time.time()
    train_loader = get_data_loader(args, shuffle=False)

    print("Delay: ", time.time() - t)
    for (sample, label) in tqdm(train_loader):
        # train_sample = np.array(train_sample)
        # sample = sample.transpose(0, 1)
        for inp in sample:
            inp = inp.unsqueeze(0)
            print(inp.size())
        print(sample.size(), label.size())
