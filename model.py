import csv
import importlib
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from utils import *


class SpeakerNet(nn.Module):
    def __init__(self, args, model, optimizer, callbacks, trainfunc, device, **kwargs):
        super(SpeakerNet, self).__init__()
        self.device = torch.device(device)
        self.args = args

        SpeakerNetModel = importlib.import_module(
            'models.' + model).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).to(self.device)

        LossFunction = importlib.import_module(
            'losses.' + trainfunc).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).to(self.device)

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.parameters(), **kwargs)
        # TODO: set up callbacks, add reduce on plateau + early stopping
        Scheduler = importlib.import_module(
            'callbacks.' + callbacks).__getattribute__('Scheduler')
        self.__scheduler__, self.lr_step = Scheduler(
            self.__optimizer__, **kwargs)
        assert self.lr_step in ['epoch', 'iteration']

    def train_network(self, loader, epoch=0):
        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        tstart = time.time()

        for i, (data, data_label) in enumerate(loader):
            data = data.transpose(0, 1)
            self.zero_grad()
            feat = []

            for inp in data:
                outp = self.__S__(inp.to(self.device))
                feat.append(outp)

            feat = torch.stack(feat, dim=1).squeeze()
            label = torch.LongTensor(data_label).to(self.device)
            nloss, prec1 = self.__L__.forward(feat, label)

            loss += nloss.detach().cpu()
            top1 += prec1
            counter += 1
            index += stepsize

            nloss.backward()
            self.__optimizer__.step()

            telapsed = time.time() - tstart
            tstart = time.time()

            sys.stdout.write(
                f"\rEpoch [{epoch}/{self.args.max_epoch}] - Processing ({index}) :")
            sys.stdout.write(
                "Loss %f TEER/TAcc %2.3f - %.2f Hz " %
                (loss / counter, top1 / counter, stepsize / telapsed))
            sys.stdout.flush()

            if self.lr_step == 'iteration':
                self.__scheduler__.step()

        if self.lr_step == 'epoch':
            self.__scheduler__.step()

        sys.stdout.write("\n")

        loss_result = loss / counter
        precision = top1 / counter

        return loss_result, precision

    def evaluateFromList(self,
                         listfilename,
                         cohorts_path='dataset/cohorts.npy',
                         print_interval=100,
                         num_eval=10,
                         eval_frames=None):

        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = None
        if cohorts_path is not None:
            cohorts = np.load(cohorts_path)

        # Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if not line:
                    break
                data = line.split()

                # Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to dictionary
        for idx, filename in enumerate(setfiles):
            inp1 = torch.FloatTensor(loadWAV(filename, eval_frames, evalmode=True, num_eval=num_eval)).to(self.device)

            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()

            feats[filename] = ref_feat
            telapsed = time.time() - tstart

            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading %d of %d: %.2f Hz, %.4f s, embedding size %d" %
                    (idx, len(setfiles), idx / telapsed, telapsed / (idx + 1), ref_feat.size()[1]))

        print('')
        all_scores = []
        all_labels = []
        all_trials = []
        tstart = time.time()

        # Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            ref_feat = feats[data[1]].to(self.device)
            com_feat = feats[data[2]].to(self.device)

            if self.__L__.test_normalize:
                ref_feat = F.normalize(ref_feat, p=2, dim=1)
                com_feat = F.normalize(com_feat, p=2, dim=1)

            # NOTE: distance(cohort = None) for training, normalized score for evaluating and testing
            if cohorts_path is None:
                dist = F.pairwise_distance(
                    ref_feat.unsqueeze(-1),
                    com_feat.unsqueeze(-1).transpose(
                        0, 2)).detach().cpu().numpy()
                score = -1 * np.mean(dist)
            else:
                score = score_normalization(ref_feat,
                                            com_feat,
                                            cohorts,
                                            top=200)

            # score = cosine_simialrity(ref_feat, com_feat)
            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz - %.4f s" %
                                 (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                sys.stdout.flush()

        print('\n')
        # # normalize scores by the largest value
        # all_scores = [float(score) / max(all_scores) for score in all_scores]
        return all_scores, all_labels, all_trials

    def testFromList(self,
                     root,
                     thre_score=0.5,
                     cohorts_path='data/zalo/cohorts.npy',
                     print_interval=100,
                     num_eval=10,
                     eval_frames=None):
        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = None
        if cohorts_path is not None:
            cohorts = np.load(cohorts_path)
        save_root = self.args.save_path + f"/{self.args.model}/result"

        data_root = Path(root, 'public_test/data_test')
        read_file = Path(root, 'public-test.csv')
        write_file = Path(save_root, 'submission_list_test.csv')
        # Read all lines from testfile (read_file)
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in tqdm(spamreader):
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to feat dictionary
        for idx, filename in enumerate(setfiles):
            inp1 = torch.FloatTensor(
                loadWAV(Path(data_root, filename),
                        eval_frames,
                        evalmode=True,
                        num_eval=num_eval)).to(self.device)
            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()
            feats[filename] = ref_feat
            telapsed = time.time() - tstart
            if idx % print_interval == 0:
                sys.stdout.write(
                    "\rReading %d of %d: %.2f Hz, %.4f s, embedding size %d" %
                    (idx, len(setfiles), (idx + 1) / telapsed, telapsed / (idx + 1), ref_feat.size()[1]))

        print('')
        tstart = time.time()

        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'label', 'score'])
            for idx, data in enumerate(lines):
                ref_feat = feats[data[0]].to(self.device)
                com_feat = feats[data[1]].to(self.device)

                if self.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    score = score_normalization(ref_feat,
                                                com_feat,
                                                cohorts,
                                                top=200)
                # score = cosine_simialrity(ref_feat, com_feat)
                pred = '1' if score >= thre_score else '0'
                spamwriter.writerow([data[0], data[1], pred, score])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz, %.4f s" %
                                     (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                    sys.stdout.flush()

            # # thresholding and write to score file
            # all_scores = [float(score) / max(all_scores)
            #               for score in all_scores]
        print('\n')

    def test_each_pair(self,
                       root,
                       thre_score=0.5,
                       cohorts_path='data/zalo/cohorts.npy',
                       print_interval=100,
                       num_eval=10,
                       eval_frames=None):
        self.eval()
        lines = []
        files = []
        pairs = []
        tstart = time.time()

        # Cohorts
        cohorts = np.load(cohorts_path)

        # Read all lines
        save_root = self.args.save_path + f"/{self.args.model}/result"
        data_root = Path(root, 'public_test/data_test')
        read_file = Path(root, 'public-test.csv')
        write_file = Path(save_root, 'submission_pair_test.csv')
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in tqdm(spamreader):
                files.append(row[0])
                files.append(row[1])
                lines.append(row)
                pairs.append([row[0], row[1]])

        setfiles = list(set(files))
        setfiles.sort()
        print("Delay time", (time.time() - tstart))
        pred_time_list = []
        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(
                ['audio_1', 'audio_2', 'label', 'inference_time'])

            for idx, pair in enumerate(pairs):
                t0 = time.time()
                path_ref = Path(data_root, pair[0])
                path_com = Path(data_root, pair[1])
                ref_feat = self.embed_utterance(path_ref,
                                                eval_frames=eval_frames,
                                                num_eval=num_eval,
                                                normalize=False).detach().cpu().to(self.device)
                com_feat = self.embed_utterance(path_com,
                                                eval_frames=eval_frames,
                                                num_eval=num_eval,
                                                normalize=False).detach().cpu().to(self.device)

                if self.__L__.test_normalize:
                    ref_feat = F.normalize(ref_feat, p=2, dim=1)
                    com_feat = F.normalize(com_feat, p=2, dim=1)

                if cohorts_path is None:
                    dist = F.pairwise_distance(
                        ref_feat.unsqueeze(-1),
                        com_feat.unsqueeze(-1).transpose(
                            0, 2)).detach().cpu().numpy()
                    score = -1 * np.mean(dist)
                else:
                    score = score_normalization(ref_feat,
                                                com_feat,
                                                cohorts,
                                                top=200)
                pred = '0'
                if score >= thre_score:
                    pred = '1'
                pred_time = time.time() - t0
                pred_time_list.append(pred_time)
                spamwriter.writerow([pair[0], pair[1], pred, pred_time])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz, %.4f s" %
                                     (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                    sys.stdout.flush()
        print('Done, avg pred time:', np.mean(pred_time_list))
        print('\n')

    def prepare(self,
                from_path='../data/test',
                save_path='checkpoints',
                prepare_type='cohorts',
                num_eval=10,
                eval_frames=0,
                print_interval=1):
        """
        Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        """
        tstart = time.time()
        self.eval()
        if prepare_type == 'cohorts':
            # Prepare cohorts for score normalization.
            # description: save all id and path of speakers to used_speaker and setfiles
            feats = []
            read_file = Path(from_path)
            files = []
            used_speakers = []
            with open(read_file, 'r') as listfile:
                while True:
                    line = listfile.readline()
                    if not line:
                        break
                    data = line.split()

                    data_1_class = Path(data[1]).parent.stem
                    data_2_class = Path(data[2]).parent.stem
                    if data_1_class not in used_speakers:
                        used_speakers.append(data_1_class)
                        files.append(data[1])
                    if data_2_class not in used_speakers:
                        used_speakers.append(data_2_class)
                        files.append(data[2])
            setfiles = list(set(files))
            setfiles.sort()

            # Save all features to file
            # desciption: load file and extrac feature embedding

            for idx, f in enumerate(tqdm(setfiles)):
                inp1 = torch.FloatTensor(
                    loadWAV(f, eval_frames, evalmode=True,
                            num_eval=num_eval)).to(self.device)

                feat = self.__S__.forward(inp1)
                if self.__L__.test_normalize:
                    feat = F.normalize(feat, p=2,
                                       dim=1).detach().cpu().numpy().squeeze()
                else:
                    feat = feat.detach().cpu().numpy().squeeze()
                feats.append(feat)

            np.save(save_path, np.array(feats))

        elif prepare_type == 'embed':
            # Prepare mean L2-normalized embeddings for known speakers.
            speaker_dirs = [x for x in Path(from_path).iterdir() if x.is_dir()]
            embeds = None
            classes = {}
            # Save mean features
            for idx, speaker_dir in enumerate(speaker_dirs):
                classes[idx] = speaker_dir.stem
                files = list(speaker_dir.glob('*.wav'))
                mean_embed = None
                embed = None
                for f in files:
                    embed = self.embed_utterance(
                        f,
                        eval_frames=eval_frames,
                        num_eval=num_eval,
                        normalize=self.__L__.test_normalize)
                    if mean_embed is None:
                        mean_embed = embed.unsqueeze(0)
                    else:
                        mean_embed = torch.cat(
                            (mean_embed, embed.unsqueeze(0)), 0)
                mean_embed = torch.mean(mean_embed, dim=0)
                if embeds is None:
                    embeds = mean_embed.unsqueeze(-1)
                else:
                    embeds = torch.cat((embeds, mean_embed.unsqueeze(-1)), -1)
                telapsed = time.time() - tstart
                if idx % print_interval == 0:
                    sys.stdout.write(
                        "\rReading %d of %d: %.4f s, embedding size %d" %
                        (idx, len(speaker_dirs), telapsed / (idx + 1), embed.size()[1]))
            print('')
            print(embeds.shape)
            # embeds = rearrange(embeds, 'n_class n_sam feat -> n_sam feat n_class')
            torch.save(embeds, Path(save_path, 'embeds.pt'))
            np.save(str(Path(save_path, 'classes.npy')), classes)
        else:
            raise NotImplementedError

    def embed_utterance(self,
                        fpath,
                        eval_frames=0,
                        num_eval=10,
                        normalize=True):
        """
        Get embedding from utterance
        """
        inp = torch.FloatTensor(
            loadWAV(fpath, eval_frames, evalmode=True,
                    num_eval=num_eval)).to(self.device)
        with torch.no_grad():
            embed = self.__S__.forward(inp)
        if normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path):
        self_state = self.state_dict()
        if self.device == 'cpu':
            loaded_state = torch.load(path, map_location=torch.device('cpu'))
        else:
            loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s" %
                      (origname, self_state[name].size(),
                       loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)
