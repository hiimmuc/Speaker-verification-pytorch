import math
import os
import csv
import argparse
import time
import random

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist

from processing.audio_loader import loadWAV, AugmentWAV
from utils import read_config
from pathlib import Path
from tqdm.auto import tqdm


def round_down(num, divisor):
    """
    To reduce number of iteration due to the increase of number of subcenters
    """
    return num - (num % divisor)

# def worker_init_fn(worker_id):
#     np.random.seed(np.random.get_state()[1][0] + worker_id)

def worker_init_fn(worker_id):
    """
    Create the init fn for worker id
    """
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2**30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % (2**30)
    np.random.seed(torch_seed + worker_id)    


class TrainLoader(Dataset):
    def __init__(self, dataset_file_name,
                 augment,
                 augment_options,
                 audio_spec,
                 aug_folder='offline', random_chunk=True, **kwargs):

        self.dataset_file_name = dataset_file_name
        self.audio_spec = audio_spec

        self.random_chunk = random_chunk
        self.max_frames = round(audio_spec['sample_rate'] * (
            audio_spec['sentence_len'] - audio_spec['win_len']) / audio_spec['hop_len'])

        self.augment = augment
        self.augment_options = augment_options

        self.sr = audio_spec['sample_rate']

        # augmented folder files
        self.aug_folder = aug_folder
        self.augment_paths = augment_options['augment_paths']
        self.augment_chain = augment_options['augment_chain']

        if self.augment and ('env_corrupt' in self.augment_chain):

            if all(os.path.exists(Path(path)) for path in self.augment_paths.values()):
                self.augment_engine = AugmentWAV(
                    augment_options, audio_spec, target_db=None)
            else:
                self.augment_engine = None
                self.augment = False

        # Read Training Files...
        lines = []
        with open(dataset_file_name, 'r', newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                # spkid, path, duration, audio_format take only 'spkid path'
                lines.append(row[:2])

        dictkeys = list(set([x[0] for x in lines]))
        dictkeys.sort()
        dictkeys = {key: ii for ii, key in enumerate(dictkeys)}

        self.label_dict = {}
        self.data_list = []
        self.data_label = []

        for lidx, data in enumerate(lines):
            speaker_label = dictkeys[data[0]]
            filename = data[1]

            self.label_dict.setdefault(speaker_label, []).append(lidx)

            self.data_label.append(speaker_label)
            self.data_list.append(filename)

    def __getitem__(self, indices):
        feat = []

        for index in indices:
            # Load audio
            audio_file = self.data_list[index]
            # time domain augment
            audio = loadWAV(audio_file, self.audio_spec,
                            evalmode=False,
                            augment=self.augment,
                            augment_options=self.augment_options,
                            random_chunk=self.random_chunk)

            # env corrupt augment
            if self.augment and ('env_corrupt' in self.augment_chain) and (self.aug_folder == 'online'):
                # if exists augmented folder(30GB) separately
                # env corruption adding from musan, revberation
                env_corrupt_proportions = self.augment_options['noise_proportion']

                augtype = np.random.choice(
                    ['rev', 'noise', 'both', 'none'], p=[0.2, 0.4, 0.2, 0.2])

                if augtype == 'rev':
                    audio = self.augment_engine.reverberate(audio)
                elif augtype == 'noise':
                    mode = np.random.choice(
                        ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                    audio = self.augment_engine.additive_noise(mode, audio)
                elif augtype == 'both':
                    # combined reverb and noise
                    order = np.random.choice(
                        ['noise_first', 'rev_first'], p=[0.5, 0.5])
                    if order == 'rev_first':
                        audio = self.augment_engine.reverberate(audio)
                        mode = np.random.choice(
                            ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                        audio = self.augment_engine.additive_noise(mode, audio)
                    else:
                        mode = np.random.choice(
                            ['noise', 'speech', 'music', 'noise_vad', 'noise_rirs'], p=env_corrupt_proportions)
                        audio = self.augment_engine.additive_noise(mode, audio)
                        audio = self.augment_engine.reverberate(audio)
                else:
                    # none type means don't augment
                    pass

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class TrainSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, distributed, seed, **kwargs):
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed

        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size

        self.data_source = data_source
        self.data_label = data_source.data_label

    def __iter__(self):

        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_label), generator=g).tolist()

        data_dict = {}

        # Sort into dictionary of file indices for each ID
        for index in indices:
            speaker_label = self.data_label[index]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = []
            data_dict[speaker_label].append(index)

        # Group file indices for each class
        dictkeys = list(data_dict.keys())
        dictkeys.sort()

        def lol(lst, sz): return [lst[i:i+sz] for i in range(0, len(lst), sz)]

        flattened_list = []
        flattened_label = []

        # Data for each class
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(
                min(len(data), self.max_seg_per_spk), self.nPerSpeaker)

            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])

        # Mix data in random order
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []

        # Prevent two pairs of the same speaker in the same batch
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)

        mixed_list = [flattened_list[i] for i in mixmap]

        # Divide data to each GPU

        if self.distributed:
            total_size = round_down(
                len(mixed_list), self.batch_size * dist.get_world_size())
            start_index = int((dist.get_rank()) /
                              dist.get_world_size() * total_size)
            end_index = int((dist.get_rank() + 1) /
                            dist.get_world_size() * total_size)
            self.num_samples = end_index - start_index
            return iter(mixed_list[start_index:end_index])
        else:
            total_size = round_down(len(mixed_list), self.batch_size)
            self.num_samples = total_size
            return iter(mixed_list[:total_size])

    def __len__(self) -> int:
        # self.num_samples
        return len(self.data_source)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


def train_data_loader(args):
    train_annotation = args.train_annotation

    batch_size = args.dataloader_options['batch_size']
    shuffle = args.dataloader_options['shuffle']
    num_workers = args.dataloader_options['num_workers']
    nPerSpeaker = args.dataloader_options['nPerSpeaker']
    max_seg_per_spk = args.dataloader_options['max_seg_per_spk']

    augment = args.augment

    train_dataset = TrainLoader(train_annotation,
                                augment,
                                args.augment_options,
                                args.audio_spec,
                                aug_folder='online')

    train_sampler = TrainSampler(
        train_dataset, nPerSpeaker=nPerSpeaker, max_seg_per_spk=max_seg_per_spk, **vars(args))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True,
        shuffle=shuffle,
    )

    return train_loader


class test_data_loader(Dataset):
    def __init__(self, test_list, audio_spec, num_eval, **kwargs):
        self.num_eval = num_eval
        self.audio_spec = audio_spec
        self.test_list = test_list

    def __getitem__(self, index):
        audio = loadWAV(self.test_list[index],
                        self.audio_spec,
                        evalmode=True,
                        augment=False,
                        augment_options=[],
                        num_eval=self.num_eval,
                        random_chunk=False)
        return torch.FloatTensor(audio), self.test_list[index]

    def __len__(self):
        return len(self.test_list)


parser = argparse.ArgumentParser(description="Data loader")
if __name__ == '__main__':
    # Test for data loader
    # YAML
    parser.add_argument('--config', type=str, default=None)

    # Device settings
    parser.add_argument('--device',
                        type=str,
                        default="cuda",
                        help='cuda or cpu')
    parser.add_argument('--distributed',
                        action='store_true',
                        default=False,
                        help='Decise wether use multi gpus')

    # Distributed and mixed precision training
    parser.add_argument('--port',
                        type=str,
                        default="8888",
                        help='Port for distributed training, input as text')
    parser.add_argument('--mixedprec',
                        dest='mixedprec',
                        action='store_true',
                        help='Enable mixed precision training')

    parser.add_argument('--augment',
                        action='store_true',
                        default=False,
                        help='Augment input')

    parser.add_argument('--early_stopping',
                        action='store_true',
                        default=False,
                        help='Early stopping')

    parser.add_argument('--seed',
                        type=int,
                        default=1000,
                        help='seed')
  #--------------------------------------------------------------------------------------#

    sys_args = parser.parse_args()

    if sys_args.config is not None:
        args = read_config(sys_args.config, sys_args)
        args = argparse.Namespace(**args)
    else:
        args = sys_args

    t = time.time()
    train_loader = train_data_loader(args)

    print("Delay: ", time.time() - t)
    print(len(train_loader))

    for (sample, label) in tqdm(train_loader):
        sample = sample.transpose(0, 1)
        print(sample.device)
        for inp in sample:
            print(inp.size(), inp.reshape(-1, inp.size()[-1]).size())
        print(sample.size(), label.size())