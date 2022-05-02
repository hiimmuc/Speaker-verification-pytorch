import csv
import importlib
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as onnxrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from processing.audio_loader import loadWAV
from utils import cprint, similarity_measure


class SpeakerNet(nn.Module):
    def __init__(self, save_path, model, optimizer, callbacks, criterion, device, max_epoch, features, **kwargs):
        super(SpeakerNet, self).__init__()
        self.device = torch.device(device)
        self.save_path = save_path
        self.model_name = model
        self.max_epoch = max_epoch
        self.scheduler_step = kwargs['scheduler_step']
        self.kwargs = kwargs
        self.T_max = 0 if 'T_max' not in kwargs else kwargs['T_max']

        SpeakerNetModel = importlib.import_module(
            'models.' + self.model_name).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).to(self.device)

        LossFunction = importlib.import_module(
            'losses.' + criterion).__getattribute__(f"{criterion}")
        self.__L__ = LossFunction(**kwargs).to(self.device)

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__(f"{optimizer}")
        self.__optimizer__ = Optimizer(self.parameters(), **kwargs)

        if features.lower() in ['mfcc', 'melspectrogram']:
            Features_extractor = importlib.import_module(
                'models.FeatureExtraction.feature').__getattribute__(f"{features.lower()}")
            self.compute_features = Features_extractor(**kwargs).to(self.device)
        else:
            Features_extractor = None
            self.compute_features = None

        # TODO: set up callbacks, add reduce on plateau + early stopping
        self.callback = callbacks
        self.lr_step = ''

        if self.callback in ['steplr', 'cosinelr', 'cycliclr']:
            Scheduler = importlib.import_module(
                'callbacks.torch_callbacks').__getattribute__(f"{callbacks.lower()}")
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **dict(kwargs, T_max=self.T_max))

            assert self.lr_step in ['epoch', 'iteration']

        elif self.callback == 'reduceOnPlateau':
            Scheduler = importlib.import_module(
                'callbacks.' + callbacks).__getattribute__('LRScheduler')
            self.__scheduler__ = Scheduler(self.__optimizer__, patience=self.scheduler_step, min_lr=1e-8, factor=0.95)

        ####
        nb_params = sum([param.view(-1).size()[0] for param in self.__S__.parameters()])
        print(f"Initialize model {self.model_name}: {nb_params:,} params")
        print("Embedding normalize: ", self.__L__.test_normalize)

    def fit(self, loader, epoch=0):
        '''Train

        Args:
            loader (Dataloader): dataloader of training data
            epoch (int, optional): [description]. Defaults to 0.

        Returns:
            tuple: loss and precision
        '''
        self.train()

        stepsize = loader.batch_size

        counter = 0
        index = 0
        loss = 0
        top1 = 0  # EER or accuracy

        loader_bar = tqdm(loader, desc=f">EPOCH_{epoch}", unit="it", colour="green")

        for (data, data_label) in loader_bar:
            data = data.transpose(0, 1)
            self.zero_grad()
            feat = []
            # forward n utterances per speaker and stack the output
            for inp in data:
                if self.compute_features is not None:
                    inp = self.compute_features(inp.to(self.device))
                outp = self.__S__(inp.to(self.device))
                feat.append(outp)

            feat = torch.stack(feat, dim=1).squeeze()
            label = torch.LongTensor(data_label).to(self.device)

            nloss, prec1 = self.__L__.forward(feat, label)

            loss += nloss.detach().cpu()
            top1 += prec1
            counter += 1
            index += stepsize

            nloss.mean().backward()
            self.__optimizer__.step()

            loader_bar.set_postfix(LR=f"{round(float(self.__optimizer__.param_groups[0]['lr']), 8)}",
                                   TLoss=f"{round(float(loss / counter), 5)}",
                                   TAcc=f"{round(float(top1 / counter), 3)}%")

            if self.lr_step == 'iteration' and self.callback in ['steplr', 'cosinelr', 'cycliclr']:
                self.__scheduler__.step()

        # select mode for callbacks
        if self.lr_step == 'epoch' and self.callback in ['steplr', 'cosinelr', 'cycliclr']:
            self.__scheduler__.step()

        elif self.callback == 'reduceOnPlateau':
            # reduceon plateau
            self.__scheduler__(loss / counter)

        elif self.callback == 'auto':
            if epoch <= 50:
                self.__scheduler__['rop'](loss / counter)
            else:
                if epoch == 51:
                    cprint("\n[INFO] # Epochs > 50, switch to steplr callback\n========>\n", 'r')
                self.__scheduler__['steplr'].step()

        loss_result = loss / (counter)
        precision = top1 / (counter)

        return loss_result, precision

    def evaluateFromList(self,
                         listfilename,
                         cohorts_path='checkpoint/dump_cohorts.npy',
                         print_interval=100,
                         num_eval=10,
                         eval_frames=None,
                         scoring_mode='cosine'):

        self.eval()

        lines = []
        files = []
        feats = {}

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
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
        print(">>>>Evaluation")

        # Save all features to dictionary
        for idx, filename in enumerate(tqdm(setfiles, desc=">>>>Reading file: ", unit="files", colour="red")):
            audio = loadWAV(filename, evalmode=True,  **self.kwargs)
            inp1 = torch.FloatTensor(audio).to(self.device)

            with torch.no_grad():
                if self.compute_features is not None:
                    inp1 = self.compute_features(inp1)
                ref_feat = self.__S__.forward(inp1.to(self.device)).detach().cpu()

            feats[filename] = ref_feat

        all_scores = []
        all_labels = []
        all_trials = []

        # Read files and compute all scores
        for idx, line in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
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
                if scoring_mode == 'norm':
                    score = similarity_measure('zt_norm',
                                               ref_feat,
                                               com_feat,
                                               cohorts,
                                               top=200)
                elif scoring_mode == 'cosine':
                    score = similarity_measure('cosine', ref_feat.cpu(), com_feat.cpu())
                elif scoring_mode == 'pnorm':
                    score = similarity_measure('pnorm', ref_feat, com_feat, p=2)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

        return all_scores, all_labels, all_trials

    def testFromList(self,
                     root,
                     test_list='evaluation_test.txt',
                     thre_score=0.5,
                     cohorts_path=None,
                     print_interval=100,
                     num_eval=10,
                     eval_frames=None,
                     scoring_mode='norm',
                     output_file=None):
        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = None
        if cohorts_path is not None and scoring_mode == 'norm':
            cohorts = np.load(cohorts_path)
        save_root = self.save_path + f"/{self.model_name}/result"

        data_root = Path(root)
        read_file = Path(test_list)
        if output_file is None:
            output_file = test_list.replace('.txt', '_result.txt')
        write_file = Path(save_root, output_file) if os.path.split(output_file)[0] == '' else output_file  # add parent dir if not provided

        # Read all lines from testfile (read_file)
        print(">>>>TESTING...")
        print(f">>> Threshold: {thre_score}")
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in spamreader:
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()

        # Save all features to feat dictionary
        for idx, filename in enumerate(tqdm(setfiles, desc=">>>>Reading file: ", unit="files", colour="red")):
            audio = loadWAV(filename.replace('\n', ''), evalmode=True, **self.kwargs)

            inp1 = torch.FloatTensor(audio).to(self.device)

            with torch.no_grad():
                if self.compute_features is not None:
                    inp1 = self.compute_features(inp1)
                ref_feat = self.__S__.forward(inp1.to(self.device)).detach().cpu()

            feats[filename] = ref_feat

        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'pred_label', 'score'])
            for idx, data in enumerate(tqdm(lines, desc=">>>>Computing files", unit="pairs", colour="MAGENTA")):
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
                    if scoring_mode == 'norm':
                        score = similarity_measure('zt_norm', ref_feat,
                                                   com_feat,
                                                   cohorts,
                                                   top=200)
                    elif scoring_mode == 'cosine':
                        score = similarity_measure('cosine', ref_feat, com_feat)
                    elif scoring_mode == 'pnorm':
                        score = similarity_measure('pnorm', ref_feat, com_feat, p=2)

                pred = '1' if score >= thre_score else '0'
                spamwriter.writerow([data[0], data[1], pred, score])

    def test_each_pair(self, root, thre_score=0.5,
                       cohorts_path='data/zalo/cohorts.npy',
                       print_interval=1,
                       num_eval=10,
                       eval_frames=None,
                       scoring_mode='norm'):
        self.eval()
        lines = []
        pairs = []
        tstart = time.time()

        # Cohorts
        cohorts = np.load(cohorts_path)

        # Read all lines
        save_root = self.save_path + f"/{self.model_name}/result"

        data_root = ''
        read_file = Path(root, 'evaluation_test.csv')
        write_file = Path(save_root, 'results_pair.txt')
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in tqdm(spamreader):
                lines.append(row)
                pairs.append([row[0], row[1]])

        print("Delay time", (time.time() - tstart))
        pred_time_list = []
        # Read files and compute all scores
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(
                ['audio_1', 'audio_2', 'label', 'inference_time', 'score'])

            for idx, pair in enumerate(pairs):
                t0 = time.time()

                score = self.pair_test(pair[0], pair[1],
                                       eval_frames,
                                       num_eval,
                                       data_root,
                                       scoring_mode,
                                       cohorts)

                pred_time = time.time() - t0
                pred_time_list.append(pred_time)

                pred = '1' if score >= thre_score else '0'

                spamwriter.writerow([pair[0], pair[1], pred, pred_time, score])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz, %.4f s" %
                                     (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                    sys.stdout.flush()
        print('Done, avg pred time:', np.mean(pred_time_list))
        print('\n')

    def pair_test(self, audio_1, audio_2, eval_frames, num_eval, data_root, scoring_mode='cosine', cohorts=None):
        assert isinstance(audio_1, str)
        assert isinstance(audio_2, str)

        audio_1 = audio_1.replace('\n', '')
        audio_2 = audio_2.replace('\n', '')

        path_ref = Path(data_root, audio_1)
        path_com = Path(data_root, audio_2)
        ref_feat = self.embed_utterance(path_ref,
                                        eval_frames=eval_frames,
                                        num_eval=num_eval,
                                        normalize=False).to(self.device)
        com_feat = self.embed_utterance(path_com,
                                        eval_frames=eval_frames,
                                        num_eval=num_eval,
                                        normalize=False).to(self.device)

        if self.__L__.test_normalize:
            ref_feat = F.normalize(ref_feat, p=2, dim=1)
            com_feat = F.normalize(com_feat, p=2, dim=1)

        if scoring_mode == 'norm':
            score = similarity_measure('zt_norm', ref_feat,
                                       com_feat,
                                       cohorts,
                                       top=200)
        elif scoring_mode == 'cosine':
            score = similarity_measure('cosine', ref_feat, com_feat)
        elif scoring_mode == 'pnorm':
            score = similarity_measure('pnorm', ref_feat, com_feat, p=2)

        return score

    def prepare(self,
                save_path=None,
                prepare_type='embed',
                num_eval=10,
                eval_frames=100,
                source=None):
        """
        Prepared 1 of the 2:
        1. Mean L2-normalized embeddings for known speakers.
        2. Cohorts for score normalization.
        """

        self.eval()
        if not source:
            raise "Please provide appropriate source!"
        # cohort preparation
        if prepare_type == 'cohorts':
            # Prepare cohorts for score normalization.
            # description: save all id and path of speakers to used_speaker and setfiles
            # source provided is a root of files -> root/spkID/spkID001.wav
            feats = []
            assert isinstance(source, str), "Please provide path to test directory"
            read_file = Path(source)
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
            # desciption: load file and extract feature embedding

            for idx, f in enumerate(tqdm(setfiles)):
                audio = loadWAV(f,
                                eval_frames,
                                evalmode=True,
                                num_eval=num_eval)

                inp1 = torch.FloatTensor(audio).to(self.device)

                if self.compute_features is not None:
                    inp1 = self.compute_features(inp1)

                feat = self.__S__.forward(inp1.to(self.device))
                if self.__L__.test_normalize:
                    feat = F.normalize(feat, p=2,
                                       dim=1).detach().cpu().numpy().squeeze()
                else:
                    feat = feat.detach().cpu().numpy().squeeze()
                feats.append(feat)
            if save_path:
                np.save(save_path, np.array(feats))
            return True

        # Embbeding preaparation
        elif prepare_type == 'embed':
            # Prepare mean L2-normalized embeddings for known speakers.
            # load audio from_path (root path)
            # option 1: from root
            if isinstance(source, str):
                speaker_dirs = [x for x in Path(source).iterdir() if x.is_dir()]
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

                print(embeds.shape)
                # embeds = rearrange(embeds, 'n_class n_sam feat -> n_sam feat n_class')
                if save_path:
                    torch.save(embeds, Path(save_path, 'embeds.pt'))
                    np.save(str(Path(save_path, 'classes.npy')), classes)
                return True

            elif isinstance(source, list):
                # option 2: list of audio in numpy format
                mean_embed = None
                embed = None
                for audio_data_np in source:
                    embed = self.embed_utterance(audio_data_np,
                                                 eval_frames=eval_frames,
                                                 num_eval=num_eval,
                                                 normalize=self.__L__.test_normalize,
                                                 sr=8000)
                    if mean_embed is None:
                        mean_embed = embed.unsqueeze(0)
                    else:
                        mean_embed = torch.cat(
                            (mean_embed, embed.unsqueeze(0)), 0)
                mean_embed = torch.mean(mean_embed, dim=0)

                if save_path:
                    torch.save(mean_embed, Path(save_path, 'embeds.pt'))
                return mean_embed
        else:
            raise NotImplementedError

    def embed_utterance(self,
                        source,
                        eval_frames=0,
                        num_eval=10,
                        normalize=False, sr=None):
        """
        Get embedding from utterance
        """
        audio = loadWAV(source,
                        eval_frames,
                        evalmode=True,
                        num_eval=num_eval,
                        sr=sr)

        inp = torch.FloatTensor(audio).to(self.device)

        with torch.no_grad():
            if self.compute_features is not None:
                inp = self.compute_features(inp)
            embed = self.__S__.forward(inp.to(self.device)).detach().cpu()
        if normalize:
            embed = F.normalize(embed, p=2, dim=1)
        return embed

    def saveParameters(self, path):
        torch.save(self.state_dict(), path)

    def loadParameters(self, path, show_error=True):
        self_state = self.state_dict()
        if self.device == torch.device('cpu'):
            loaded_state = torch.load(path, map_location=torch.device('cpu'))
        else:
            print(f"Load model in {torch.cuda.device_count()} GPU(s)")
            loaded_state = torch.load(path, map_location=torch.device(self.device))
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")

                if name not in self_state:
                    if show_error:
                        print("%s is not in the model." % origname)
                    continue

            if self_state[name].size() != loaded_state[origname].size():
                if show_error:
                    print("Wrong parameter length: %s, model: %s, loaded: %s" %
                          (origname, self_state[name].size(),
                           loaded_state[origname].size()))
                continue

            self_state[name].copy_(param)

    def export_onnx(self, state_path, check=True):
        save_root = self.save_path + f"/{self.model_name}/model"
        save_path = os.path.join(save_root, f"model_eval_{self.model_name}.onnx")

        input_names = ["input"]
        output_names = ["output"]
        # NOTE: because using torch.audio -> cant export onnx
        dummy_input = torch.randn(10, 8120, device="cuda")

        self.loadParameters(state_path)
        self.__S__.eval()

        torch.onnx.export(self.__S__,
                          dummy_input,
                          save_path,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names,

                          export_params=True,
                          opset_version=11)

        # double check
        if os.path.exists(save_path) and check:
            print("checking export")
            model = onnx.load(save_path)
            onnx.checker.check_model(model)
            print(onnx.helper.printable_graph(model.graph))
            cprint("Done!!!", 'r')

    def onnx_inference(self, model_path, inp):
        def to_numpy(tensor):
            if not torch.is_tensor(tensor):
                tensor = torch.FloatTensor(tensor)
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        onnx_session = onnxrt.InferenceSession(model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: to_numpy(inp)}
        onnx_output = onnx_session.run(None, onnx_inputs)
        return onnx_output
##################################################################################################
