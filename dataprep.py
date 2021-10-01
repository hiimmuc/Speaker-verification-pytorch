# run on linux only
import argparse
import glob
import hashlib
import os
import random
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from zipfile import ZipFile

import librosa
import numpy as np
import scipy
import soundfile as sf
from scipy.io import wavfile
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm.auto import tqdm

import config as cfg
from utils import *


def get_audio_path(folder):
    """
    Get the audio path for a given folder

    Args:
        folder ([type]): [description]

    Returns:
        list: [description]
    """
    return glob.glob(os.path.join(folder, '*.wav'))


def md5(fname):
    """
    MD5SUM
    """
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def download(args, lines):
    """
    Download with wget
    """
    for line in lines:
        url = line.split()[0]
        md5gt = line.split()[1]
        outfile = url.split('/')[-1]

        # Download files
        out = subprocess.call('wget %s -O %s/%s' %
                              (url, args.save_path, outfile),
                              shell=True)
        if out != 0:
            raise ValueError(
                'Download failed %s. If download fails repeatedly, use alternate URL on the VoxCeleb website.'
                % url)

        # Check MD5
        md5ck = md5('%s/%s' % (args.save_path, outfile))
        if md5ck == md5gt:
            print('Checksum successful %s.' % outfile)
        else:
            raise Warning('Checksum failed %s.' % outfile)


def full_extract(args, fname):
    """
    Extract zip files
    """
    print('Extracting %s' % fname)
    if fname.endswith(".tar.gz"):
        with tarfile.open(fname, "r:gz") as tar:
            tar.extractall(args.save_path)
    elif fname.endswith(".zip"):
        with ZipFile(fname, 'r') as zf:
            zf.extractall(args.save_path)


def part_extract(args, fname, target):
    """
    Partially extract zip files
    """
    print('Extracting %s' % fname)
    with ZipFile(fname, 'r') as zf:
        for infile in zf.namelist():
            if any([infile.startswith(x) for x in target]):
                zf.extract(infile, args.save_path)


def split_musan(args):
    """
    Split MUSAN for faster random access
    """

    files = glob.glob('%s/musan/*/*/*.wav' % args.augment_path)

    audlen = 16000 * 5
    audstr = 16000 * 3

    for idx, file in enumerate(files):
        fs, aud = wavfile.read(file)
        writedir = os.path.splitext(file.replace('/musan/',
                                                 '/musan_split/'))[0]
        os.makedirs(writedir)
        for st in range(0, len(aud) - audlen, audstr):
            wavfile.write(writedir + '/%05d.wav' % (st / fs), fs,
                          aud[st:st + audlen])

        print(idx, file)


def prepare_augmentation(args):
    """
    Check wether the augmentation dataset is already downloaded

    Args:
        args ([type]): [description]
    """
    # TODO: check if the augmentation dataset is already downloaded -> extract only
    if not os.path.exists(args.augment_path):
        print('Downloading augmentation dataset...')
        with open('dataset/augment.txt', 'r') as f:
            augfiles = f.readlines()
        download(args, augfiles)

        part_extract(args, os.path.join(args.augment_path, 'rirs_noises.zip'), [
            'RIRS_NOISES/simulated_rirs/mediumroom',
            'RIRS_NOISES/simulated_rirs/smallroom'
        ])

        full_extract(args, os.path.join(args.augment_path, 'musan.tar.gz'))

        split_musan(args)
    else:
        print('Augmentation dataset already exists in', args.augment_path)

    if not os.path.exists(args.raw_dataset):
        raise "Raw dataset is empty"


def augmentation(args, audio_paths, mode='train', max_frames=cfg.mfcc_config.max_samples, step_save=500):
    """
    Perfrom augmentation on the raw dataset
    """

    prepare_augmentation(args)  # check if augumentation data is ready

    aug_rate = args.aug_rate
    musan_path = Path(args.augment_path, 'musan_split')
    rir_path = Path(args.augment_path, 'RIRS_NOISES/simulated_rirs')
    print('Start augmenting data with', musan_path, 'and', rir_path)

    if mode == 'train':
        print('Augment Full')
        num_aug = len(audio_paths)
        augment_audio_paths = audio_paths
    elif mode == 'test':
        num_aug = int(aug_rate * len(audio_paths))
        random_indices = random.sample(range(len(audio_paths)), num_aug)
        augment_audio_paths = [audio_paths[i] for i in random_indices]
    else:
        raise ValueError('mode should be train or test')

    print('Number of augmented data: {}/{}'.format(num_aug, len(audio_paths)))

    augment_engine = AugmentWAV(musan_path, rir_path, max_frames)

    list_audio = []

    for idx, fpath in enumerate(tqdm(augment_audio_paths, unit='files', desc=f"Augmented process")):
        audio, sr = loadWAV(fpath, max_frames=max_frames,
                            evalmode=False, sr=16000)
        if mode == 'test':
            aug_type = random.randint(1, 4)

            if aug_type == 1:
                audio = augment_engine.reverberate(audio)
            elif aug_type == 2:
                audio = augment_engine.additive_noise('music', audio)
            elif aug_type == 3:
                audio = augment_engine.additive_noise('speech', audio)
            elif aug_type == 4:
                audio = augment_engine.additive_noise('noise', audio)

            list_audio.append([audio, aug_type])
            s = 1
        else:
            aug_audio1 = augment_engine.reverberate(audio)
            aug_audio2 = augment_engine.additive_noise('music', audio)
            aug_audio3 = augment_engine.additive_noise('speech', audio)
            aug_audio4 = augment_engine.additive_noise('noise', audio)
            aug_audio = [aug_audio1, aug_audio2, aug_audio3, aug_audio4]

            for i, audio_ in enumerate(aug_audio):
                list_audio.append([audio_, i + 1])
            s = 4

        roots = [os.path.split(fpath)[0] for fpath in augment_audio_paths]
        # change_path to test folder
        if mode == 'test':
            roots = [path.replace('wavs', 'test') for path in roots]
        audio_names = [os.path.split(fpath)[1]
                       for fpath in augment_audio_paths]

        # save list of augment audio each step = save_step (default 500)
        if (idx + 1) % step_save == 0 or (idx == len(augment_audio_paths) - 1):
            ii = ((idx + 1) // step_save - 1) * \
                step_save if idx + 1 >= step_save else 0

            for i, (audio, aug_t) in enumerate(tqdm(list_audio, unit='file', desc=f'Save augmented files {ii} -> {idx}')):
                save_path = os.path.join(
                    roots[i//s + ii], f"{audio_names[i//s + ii].replace('.wav', '')}_augmented_{aug_t}.wav")

                if os.path.exists(save_path):
                    print(f"overwrite {idx} to id {i//s + ii}")
                    continue
                else:
                    os.makedirs(os.path.split(save_path)[0], exist_ok=True)
                # NOTE: still have error, duplicate files
                # shape: (channel, frames) -> (frames, channels)
                audio = audio.T
                sf.write(str(save_path), audio, sr)
            list_audio = []  # refresh list to avoid memory overload
    print('Done!')


def clean_dump_files(args):
    """check whether the structure is not correct"""
    data_files = []
    raw_path = args.raw_dataset
    with open(os.path.join(args.save_dir, 'data_folders.txt'), 'r') as f:
        data_files = f.readlines()
        data_files = list(
            map(lambda x: x.replace('\n', ''), data_files))
    for path in tqdm(data_files):
        for invalid in os.listdir(path):
            path_invalid = os.path.join(path, invalid)
            if os.path.isdir(path_invalid):
                print(path_invalid, end=' ')
                if len(os.listdir(path_invalid)) == 0:
                    print('empty', end='\n')
                    os.rmdir(path_invalid)  # remove empty folder
                else:
                    # move file to parent folder
                    if os.path.isdir(os.path.join(raw_path, invalid)):
                        for audio in os.listdir(path_invalid):
                            if audio not in os.listdir(os.path.join(raw_path, invalid)):
                                shutil.move(src=os.path.join(path_invalid, audio),
                                            dst=os.path.join(os.path.join(raw_path, invalid), audio))
                        shutil.rmtree(path_invalid)


class FeatureExtraction:
    def __init__(self, args):
        self.config = cfg.mfcc_config
        self.data_files = []
        self.args = args
        self.save_dir = args.save_dir
        self.get_labels('train')
        self.feat_save_dir = f'./dataset/feature_vectors/{time.strftime("%m-%d-%H-%M", time.localtime())}'
        os.makedirs(self.feat_save_dir, exist_ok=True)

    def get_labels(self, dts):
        with open(os.path.join(self.save_dir, f"{dts}.txt"), 'r') as f:
            self.data_files = f.readlines()
        self.data_files = list(
            map(lambda x: x.replace('\n', '').split(' ')[1], self.data_files))
        # get label of audio
        self.data_labels = list(
            map(lambda x: x.replace('\n', '').split(' ')[0], self.data_files))

    def extract_feature_frame(self, audio_path):
        '''Extract Mel spectrogram features from audio file

        Args:
            audio_path (str or file type): path to audio file

        Returns:
            ndarray: mfccs (frames, channels)
            sampling_rate: int
        '''
        try:

            audio, sample_rate = librosa.load(
                audio_path, sr=self.config.sampling_rate)
            y = audio

            # calculate by avg time length / time overlap - 1 for st :v
            max_pad_length = self.config.max_pad_length
            win_length = self.config.max_pad_length
            n_fft = self.config.n_fft
            hop_length = self.config.hop_length
            n_mels = self.config.n_mels
            window = scipy.signal.hamming(win_length, sym=False)

            mfccs = librosa.feature.melspectrogram(y=y, sr=sample_rate,
                                                   n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   win_length=win_length,
                                                   n_mels=n_mels,
                                                   window=window)

            pad_width = max_pad_length - mfccs.shape[1]
            pad_width = pad_width if pad_width >= 0 else 0
            mfccs = np.pad(mfccs[:, :max_pad_length], pad_width=(
                (0, 0), (0, pad_width)), mode='constant')

        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None, self.config.sampling_rate
        return mfccs, sample_rate

    def process_raw_dataset(self):
        """
        Load and preprocess the audio files
        """
        print("Start text-independent utterance feature extraction")
        if not isinstance(self.data_files, list):
            print("wrong format of folder path")
            return None, None
        total_speaker_num = len(self.data_files)
        print("Total speaker files : %d" % total_speaker_num)
        features = []
        labels = []
        for i, audio_path in enumerate(tqdm(self.data_files, unit='files', desc='Extracting features')):
            # same speaker -> same label
            label = self.data_labels[i]

            feat, _ = self.extract_feature_frame(audio_path=audio_path)
            if feat is not None:
                features.append(feat)
                labels.append(label)
            else:
                print("Data is empty: ", audio_path)
                continue

        X = np.array(features)
        Y = np.array(labels)

        # one hot label
        y_l = Y.copy()
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(Y)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        Y = np.asanyarray(onehot_encoded)
        return X, Y, y_l, np.asanyarray(integer_encoded)

    def save_as_ndarray(self, X, y_onehot, y_raw, y_int_encode):
        print('Saving into npy format')
        np.save(os.path.join(self.feat_save_dir, "data.npy"), X)
        np.save(os.path.join(self.feat_save_dir, "label_onehot.npy"), y_onehot)
        np.save(os.path.join(self.feat_save_dir, "label_raw.npy"), y_raw)
        np.save(os.path.join(self.feat_save_dir,
                "label_int_encode.npy"), y_int_encode)
        print('Done!')


class DataGenerator():
    def __init__(self, args, **kwargs):
        self.args = args
        self.data_paths = self.get_data_paths()
        if self.args.convert:
            clean_dump_files(self.args)

    def get_data_paths(self):
        raw_data_dir = self.args.raw_dataset

        data_paths = []
        for fdir in os.listdir(raw_data_dir):
            data_paths.extend(
                glob.glob(os.path.join(raw_data_dir, f'{fdir}/*.wav')))

        with open(os.path.join(self.args.save_dir, 'data.txt'), 'w') as f:
            for path in data_paths:
                f.write(f'{path}\n')
        data_folder = list(set([os.path.split(path)[0]
                           for path in data_paths]))

        with open(os.path.join(self.args.save_dir, 'data_folders.txt'), 'w') as f:
            for path in data_folder:
                f.write(f'{path}\n')
        non_augment_path = list(
            filter(lambda x: 'augment' not in str(x), data_paths))
        augment_data_paths = list(filter(lambda x: 'augment' in str(x), data_paths))
        return non_augment_path, augment_data_paths

    def convert(self):
        # convert data to one form 16000Hz, only works on Linux
        files = list(Path(self.args.raw_dataset).glob('*/*.wav'))
        files.sort()
        print('Converting files, Total:', len(files))
        for fpath in tqdm(files):
            fpath = str(fpath).replace('(', '\(')
            fpath = fpath.replace(')', '\)')
            outpath = fpath[:-4] + '_conv' + fpath[-4:]
            out = subprocess.call(
                'ffmpeg -y -i %s -ac 1 -vn -acodec pcm_s16le -ar 16000 %s >/dev/null 2>/dev/null'
                % (fpath, outpath),
                shell=True)
            if out != 0:
                raise ValueError('Conversion failed %s.' % fpath)
            subprocess.call('rm %s' % (fpath), shell=True)
            subprocess.call('mv %s %s' % (outpath, fpath), shell=True)
        print('Done!')

    def generate_lists(self):
        """
        Generate train test lists for zalo data
        """
        root = Path(self.args.raw_dataset)
        train_writer = open(Path(root.parent, 'train.txt'), 'w')
        val_writer = open(Path(root.parent, 'val.txt'), 'w')
        classpaths = [d for d in root.iterdir() if d.is_dir()]
        print('Generate dataset metadata files, total:', len(classpaths))
        val_filepaths_list = []
        for classpath in classpaths:
            filepaths = list(classpath.glob('*.wav'))

            random.shuffle(filepaths)

            non_augment_path = list(
                filter(lambda x: 'augment' not in str(x), filepaths))

            val_num = 3  # 3 utterances per speaker for val

            if self.args.split_ratio > 0:
                val_num = int(self.args.split_ratio * len(non_augment_path))

            val_filepaths = non_augment_path[:val_num]
            train_filepaths = non_augment_path[val_num:]

            for train_filepath in train_filepaths:
                label = str(train_filepath.parent.stem.split('-')[0])
                train_writer.write(label + ' ' + str(train_filepath) + '\n')
            val_filepaths_list.append(val_filepaths)

        for val_filepaths in val_filepaths_list:
            for i in range(len(val_filepaths) - 1):
                for j in range(i + 1, len(val_filepaths)):
                    label = '1'
                    val_writer.write(label + ' ' + str(val_filepaths[i]) + ' ' +
                                     str(val_filepaths[j]) + '\n')
                    label = '0'
                    while True:
                        x = random.randint(0, len(val_filepaths_list) - 1)
                        if not val_filepaths_list[x]:
                            continue
                        if val_filepaths_list[x][0].parent.stem != val_filepaths[
                                i].parent.stem:
                            break
                    y = random.randint(0, len(val_filepaths_list[x]) - 1)
                    val_writer.write(label + ' ' + str(val_filepaths[i]) + ' ' +
                                     str(val_filepaths_list[x][y]) + '\n')

        train_writer.close()
        val_writer.close()

    def transform(self):
        """Transform dataset from raw wave to compressed numpy array"""
        feat_extract_engine = FeatureExtraction(self.args)
        data = feat_extract_engine.process_raw_dataset()
        feat_extract_engine.save_as_ndarray(data[0], data[1], data[2], data[3])


parser = argparse.ArgumentParser(description="Data preparation")
if __name__ == '__main__':
    parser.add_argument('--save_dir',
                        type=str,
                        default="dataset/",
                        help='Directory to save files(parent root)')
    parser.add_argument('--raw_dataset',
                        type=str,
                        default="dataset/wavs",
                        help='Deractory consists raw dataset')

    parser.add_argument('--split_ratio',
                        type=float,
                        default=0.2,
                        help='Split ratio')
    # mode
    parser.add_argument('--convert',
                        default=False,
                        action='store_true',
                        help='Enable coversion')
    parser.add_argument('--generate',
                        default=False,
                        action='store_true',
                        help='Enable generate')
    parser.add_argument('--transform',
                        default=False,
                        action='store_true',
                        help='Enable transformation')
    # augmentation
    parser.add_argument('--augment',
                        default=False,
                        action='store_true',
                        help='Download and extract augmentation files')
    parser.add_argument('--augment_mode',
                        type=str,
                        default='train',
                        help='')
    parser.add_argument('--augment_path',
                        type=str,
                        default='dataset/augment_data',
                        help='Directory include augmented data')
    parser.add_argument('--aug_rate',
                        type=float,
                        default=0.5,
                        help='')

    args = parser.parse_args()

    data_generator = DataGenerator(args)
    print('Start processing...')

    if args.augment:
        augmentation(
            args=args, audio_paths=data_generator.data_paths[0][:], step_save=200, mode=args.augment_mode)
    if args.convert:
        data_generator.convert()
    if args.generate:
        data_generator.generate_lists()
    if args.transform:
        data_generator.transform()

# TODO: 10 ddiem du lieu bij loi luu tru 10601 -> 10610
