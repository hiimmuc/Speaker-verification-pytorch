import argparse
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from processing.audio_loader import AugmentWAV, loadWAV
from processing.vad_tool import VAD
from utils import round_down, worker_init_fn


class Loader(Dataset):
    def __init__(self, dataset_file_name, augment, musan_path, rir_path, max_frames, n_mels, aug_folder='offline', **kwargs):

        self.dataset_file_name = dataset_file_name
        self.max_frames = max_frames
        self.augment = augment
        self.n_mels = n_mels
        self.kwargs = kwargs
        self.sr = kwargs['sample_rate']

        # augmented folder files
        self.aug_folder = aug_folder
        self.musan_path = musan_path
        self.rir_path = rir_path
        self.augment_chain = kwargs['augment_chain'] if 'augment_chain' in kwargs else ['env_corrupt', 'time_domain']

        if self.augment and ('env_corrupt' in self.augment_chain):
            if all(os.path.exists(path) for path in [self.musan_path, self.rir_path]):
                self.augment_engine = AugmentWAV(musan_path=musan_path,
                                                 rir_path=rir_path,
                                                 max_frames=max_frames,
                                                 sample_rate=self.sr,
                                                 target_db=None)
            else:
                self.augment_engine = None

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

            if not (speaker_label in self.label_dict):
                self.label_dict[speaker_label] = []

            self.label_dict[speaker_label].append(lidx)
            self.data_label.append(speaker_label)
            self.data_list.append(data[1])

    def __getitem__(self, indices):
        feat = []

        for index in indices:
            # Load audio
            audio_file = self.data_list[index]
#             if self.augment and ('env_corrupt' in self.augment_chain) and (self.aug_folder == 'offline'):
#                 # if aug audio files and raw audio files is in the same folder
#                 aug_type = random.randint(0, 4)
#                 if aug_type == 0:
#                     pass
#                 else:
#                     # for type 1 -> 4
#                     audio_file = f"{self.data_list[index].replace('.wav', '')}_augmented_{aug_type}.wav"

            # time domain augment
            audio = loadWAV(audio_file, self.max_frames,
                            evalmode=False,
                            augment=self.augment,
                            sample_rate=self.sr,
                            augment_chain=self.augment_chain)

            # env corrupt augment
            if self.augment and ('env_corrupt' in self.augment_chain) and (self.aug_folder == 'online'):
                # if exists augmented folder(30GB) separately
                # env corruption adding from musan, revberation
                augtype = np.random.choice(['rev', 'noise', 'both', 'none'], p=[0.2, 0.4, 0.2, 0.2])
                if augtype == 'rev':
                    audio = self.augment_engine.reverberate(audio)
                elif augtype == 'noise':
                    mode = np.random.choice(['music', 'speech', 'noise', 'noise_vad'], p=[0.5, 0, 0.5, 0])
                    audio = self.augment_engine.additive_noise(mode, audio)
                elif augtype == 'both':
                    # combined reverb and noise
                    order = np.random.choice(['noise_first', 'rev_first'], p=[0.5, 0.5])
                    if order == 'rev_first':
                        audio = self.augment_engine.reverberate(audio)
                        mode = np.random.choice(['music', 'speech', 'noise', 'noise_vad'], p=[0.5, 0, 0.5, 0])
                        audio = self.augment_engine.additive_noise(mode, audio)
                    else:
                        mode = np.random.choice(['music', 'speech', 'noise', 'noise_vad'], p=[0.5, 0, 0.5, 0])
                        audio = self.augment_engine.additive_noise(mode, audio)
                        audio = self.augment_engine.reverberate(audio)
                else:
                    # none type means dont augment
                    pass

            feat.append(audio)

        feat = np.concatenate(feat, axis=0)

        return torch.FloatTensor(feat), self.data_label[index]

    def __len__(self):
        return len(self.data_list)


class Sampler(torch.utils.data.Sampler):
    def __init__(self, data_source, nPerSpeaker, max_seg_per_spk, batch_size, **kwargs):
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


def get_data_loader(dataset_file_name, batch_size, augment, musan_path,
                    rir_path, max_frames, max_seg_per_spk, nDataLoaderThread,
                    nPerSpeaker, n_mels, **kwargs):
    train_dataset = Loader(dataset_file_name, augment, musan_path,
                           rir_path, max_frames, n_mels, aug_folder='online', **kwargs)

    train_sampler = Sampler(train_dataset, nPerSpeaker,
                            max_seg_per_spk, batch_size, **kwargs)

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


# Test Data Loader
parser = argparse.ArgumentParser(description="Data loader")
if __name__ == '__main__':
    # Test for data loader
    parser.add_argument('--augment',
                        action='store_true',
                        default=True,
                        help='decide whether use augment data')
    parser.add_argument('--train_list',
                        type=str,
                        default="dataset/train_callbot_v2/train_def.txt",
                        help='Directory to save files(parent root)')
    parser.add_argument('--batch_size',
                        type=int,
                        default=32,
                        help='Batch size, number of speakers per batch')
    parser.add_argument('--max_seg_per_spk',
                        type=int,
                        default=100,
                        help='Maximum number of utterances per speaker per epoch')
    parser.add_argument('--nDataLoaderThread',
                        type=int,
                        default=0,
                        help='Number of loader threads')
    parser.add_argument('--nPerSpeaker',
                        type=int,
                        default=2,
                        help='Number of utterances per speaker per batch, only for metric learning based losses')
    parser.add_argument('--max_frames',
                        type=int,
                        default=100,
                        help='Input length to the network for training, 1s ~ 100 frames')
    parser.add_argument('--musan_path',
                        type=str,
                        default="dataset/augment_data/musan_split",
                        help='Absolute path to the augment set')
    parser.add_argument('--rir_path',
                        type=str,
                        default="dataset/augment_data/RIRS_NOISES/simulated_rirs",
                        help='Absolute path to the augment set')

    # Model definition for MFCCs
    parser.add_argument('--n_mels',
                        type=int,
                        default=80,
                        help='Number of mel filter banks')

    parser.add_argument('--sample_rate',
                        type=int,
                        default=8000,
                        help='Number of sample_rate')
    args = parser.parse_args()
    t = time.time()
    train_loader = get_data_loader(args.train_list, **vars(args))

    print("Delay: ", time.time() - t)
    print(len(train_loader))
    for (sample, label) in tqdm(train_loader):
        sample = sample.transpose(0, 1)
        for inp in sample:
            print(inp.size())
        print(sample.size(), label.size())
