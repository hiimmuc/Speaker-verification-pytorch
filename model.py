import csv
import importlib
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

from utils import *


class SpeakerNet(nn.Module):
    def __init__(self, save_path, model, optimizer, callbacks, criterion, device, max_epoch, preprocess, **kwargs):
        super(SpeakerNet, self).__init__()
        self.device = torch.device(device)
        self.save_path = save_path
        self.model_name = model
        self.max_epoch = max_epoch
        self.apply_preprocess = preprocess

        SpeakerNetModel = importlib.import_module(
            'models.' + self.model_name).__getattribute__('MainModel')
        self.__S__ = SpeakerNetModel(**kwargs).to(self.device)
        nb_params = sum([param.view(-1).size()[0] for param in self.__S__.parameters()])
        print(f"Initialize model {self.model_name}: {nb_params} params")

        LossFunction = importlib.import_module(
            'losses.' + criterion).__getattribute__('LossFunction')
        self.__L__ = LossFunction(**kwargs).to(self.device)

        Optimizer = importlib.import_module(
            'optimizer.' + optimizer).__getattribute__('Optimizer')
        self.__optimizer__ = Optimizer(self.parameters(), **kwargs)

        # TODO: set up callbacks, add reduce on plateau + early stopping
        self.callback = callbacks
        self.lr_step = ''
        if self.callback in ['steplr', 'cosinelr']:
            Scheduler = importlib.import_module(
                'callbacks.' + callbacks).__getattribute__('Scheduler')
            self.__scheduler__, self.lr_step = Scheduler(self.__optimizer__, **kwargs)

            assert self.lr_step in ['epoch', 'iteration']

        elif self.callback == 'reduceOnPlateau':
            Scheduler = importlib.import_module(
                'callbacks.' + callbacks).__getattribute__('LRScheduler')
            self.__scheduler__ = Scheduler(self.__optimizer__, patience=8, min_lr=1e-6, factor=0.9)

        elif self.callback == 'auto':
            steplrScheduler = importlib.import_module('callbacks.' + 'steplr').__getattribute__('Scheduler')
            ropScheduler = importlib.import_module('callbacks.' + 'reduceOnPlateau').__getattribute__('LRScheduler')

            self.__scheduler__ = {}
            self.__scheduler__['steplr'], self.lr_step = steplrScheduler(self.__optimizer__, **kwargs)
            self.__scheduler__['rop'] = ropScheduler(self.__optimizer__, patience=8, min_lr=1e-6, factor=0.9)

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
        
        tstart = time.time()
        
        loader_bar = tqdm(loader, desc=f">>>EPOCH {epoch}: ", unit="it", colour="green")
        for (data, data_label) in loader_bar:
            data = data.transpose(0, 1)
            self.zero_grad()
            feat = []
            # forward n utterances per speaker and stack the output
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
            loader_bar.set_postfix(TLoss=f"{round(float(loss / counter), 5)}", TAcc=f"{round(float(top1 / counter), 3)}%")

#             telapsed = time.time() - tstart
#             tstart = time.time()

#             sys.stdout.write(
#                 f"\rEpoch [{epoch}/{self.max_epoch}] - Processing ({index}): ")
#             sys.stdout.write(
#                 "Loss %f TEER/TAcc %2.3f - %.2f Hz " %
#                 (loss / counter, top1 / counter, stepsize / telapsed))
#             sys.stdout.flush()

            if self.lr_step == 'iteration' and self.callback in ['steplr', 'cosinelr']:
                self.__scheduler__.step()

        # select mode for callbacks
        if self.lr_step == 'epoch' and self.callback in ['steplr', 'cosinelr']:
            self.__scheduler__.step()

        elif self.callback == 'reduceOnPlateau':
            # reduceon plateau
            self.__scheduler__(loss / counter)

        elif self.callback == 'auto':
            if epoch <= 50:
                self.__scheduler__['rop'](loss / counter)
            else:
                if epoch == 51:
                    print("\n[INFO] # epochs > 100, switch to steplr callback")
                self.__scheduler__['steplr'].step()

#         sys.stdout.write("\n")

        loss_result = loss / (counter)
        precision = top1 / (counter)

        return loss_result, precision

    def evaluateFromList(self,
                         listfilename,
                         cohorts_path='dataset/cohorts.npy',
                         print_interval=100,
                         num_eval=10,
                         eval_frames=None,
                         scoring_mode='norm'):

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
        print(">>>>Evaluation")

        # Save all features to dictionary
        for idx, filename in enumerate(tqdm(setfiles, desc=">>>>Reading file: ", unit="it", colour="red")):
            audio = loadWAV(filename, eval_frames, evalmode=True, num_eval=num_eval)
            if self.apply_preprocess:
                audio = mels_spec_preprocess(audio)

            inp1 = torch.FloatTensor(audio).to(self.device)

            with torch.no_grad():
                ref_feat = self.__S__.forward(inp1).detach().cpu()

            feats[filename] = ref_feat
            # telapsed = time.time() - tstart

#             if idx % print_interval == 0:
#                 sys.stdout.write(
#                     "\rReading %d of %d: %.2f Hz, %.4f s, embedding size %d" %
#                     (idx, len(setfiles), idx / telapsed, telapsed / (idx + 1), ref_feat.size()[1]))

#         print('')
        all_scores = []
        all_labels = []
        all_trials = []
        # tstart = time.time()

        # Read files and compute all scores

            
        for idx, line in enumerate(tqdm(lines, desc=">>>>Computing files", unit="it", colour="MAGENTA")):
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
                    score = score_normalization(ref_feat,
                                                com_feat,
                                                cohorts,
                                                top=200)
                elif scoring_mode == 'cosine':
                    score = cosine_simialrity(ref_feat, com_feat)

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])             
                

#             if idx % print_interval == 0:
#                 telapsed = time.time() - tstart
#                 sys.stdout.write("Computing %d of %d: %.2f Hz - %.4f s" %
#                                  (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
#                 sys.stdout.flush()

#         print('\n')
        return all_scores, all_labels, all_trials

    def testFromList(self,
                     root,
                     thre_score=0.5,
                     cohorts_path='data/zalo/cohorts.npy',
                     test_set='public',
                     print_interval=100,
                     num_eval=10,
                     eval_frames=None,
                     scoring_mode='norm'):
        self.eval()

        lines = []
        files = []
        feats = {}
        tstart = time.time()

        # Cohorts
        cohorts = None
        if cohorts_path is not None:
            cohorts = np.load(cohorts_path)
        save_root = self.save_path + f"/{self.model_name}/result"

        data_root = Path(root)
        read_file = Path(root, 'evaluation_test.txt')
        write_file = Path(save_root, 'results.txt')
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
            audio = loadWAV(filename.replace('\n', ''),
                            eval_frames,
                            evalmode=True,
                            num_eval=num_eval)

            if self.apply_preprocess:
                audio = mels_spec_preprocess(audio)

            inp1 = torch.FloatTensor(audio).to(self.device)

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
                    if scoring_mode == 'norm':
                        score = score_normalization(ref_feat,
                                                    com_feat,
                                                    cohorts,
                                                    top=200)
                    elif scoring_mode == 'cosine':
                        score = cosine_simialrity(ref_feat, com_feat)

                pred = '1' if score >= thre_score else '0'
                spamwriter.writerow([data[0], data[1], pred, score])

                if idx % print_interval == 0:
                    telapsed = time.time() - tstart
                    sys.stdout.write("\rComputing %d of %d: %.2f Hz, %.4f s" %
                                     (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                    sys.stdout.flush()

        print('\n')

    def test_each_pair(self,
                       root,
                       thre_score=0.5,
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
            score = score_normalization(ref_feat,
                                        com_feat,
                                        cohorts,
                                        top=200)
        elif scoring_mode == 'cosine':
            score = cosine_simialrity(ref_feat, com_feat)

        return score

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
                audio = loadWAV(f,
                                eval_frames,
                                evalmode=True,
                                num_eval=num_eval)

                if self.apply_preprocess:
                    audio = mels_spec_preprocess(audio)

                inp1 = torch.FloatTensor(audio).to(self.device)

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

        if self.apply_preprocess:
            audio = mels_spec_preprocess(audio)

        inp = torch.FloatTensor(audio).to(self.device)

        with torch.no_grad():
            embed = self.__S__.forward(inp).detach().cpu()
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
        save_root = self.save_path + f"/{self.model}/model"
        save_path = os.path.join(save_root, f"model_{self.model}.onnx")

        input_names = ["input"]
        output_names = ["output"]
        # NOTE: because using torch.audio -> cant export onnx
        dummy_input = torch.FloatTensor(
            loadWAV(r"dataset\wavs\5-F-27\5-3.wav", 100, evalmode=True,
                    num_eval=10)).to(self.device)

        if self.apply_preprocess:
            dummy_input = mels_spec_preprocess(dummy_input)

        self.loadParameters(state_path)
        self.__S__.eval()

        torch.onnx.export(self.__S__,
                          dummy_input,
                          save_path,
                          verbose=False,
                          input_names=input_names,
                          output_names=output_names,
                          export_params=True)

        # double check
        if os.path.exists(save_path) and check:
            model = onnx.load(save_path)
            onnx.checker.check_model(model)
            onnx.helper.printable_graph(model.graph)

    def onnx_inference(self, model_path, inp):
        onnx_session = onnxrt.InferenceSession(model_path)
        onnx_inputs = {onnx_session.get_inputs()[0].name: torch.to_numpy(inp)}
        onnx_output = onnx_session.run(None, onnx_inputs)
        return onnx_output
    
    def export_folder(self, folder_path):
        model_saved_path = os.path.join(folder_path, self.model_name)
        os.makedirs(model_saved_path, exist_ok=True)
        
        saved_state = os.path.join(model_saved_path, 'best_state.model')
        saved_model = os.path.join(model_saved_path, 'best_model.pt')
        
        torch.save(self.state_dict(), saved_state)
        torch.save(self.__S__, saved_model)
        
        config_eval_file = os.path.join(model_saved_path, "config_eval.yaml")
        config_deploy_file = os.path.join(model_saved_path, "config_deploy.yaml")

        # with open()
        # save all to 1 file
        pass