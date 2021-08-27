import random
import sys
from math import fabs
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SpeakerNet
from utils import tuneThresholdfromScore


def inference(args):
    model = SpeakerNet(**vars(args))
    model.loadParameters(args.initial_model_infer)
    model.eval()
    # cohorts = np.load('checkpoints/cohorts_final_500_f100.npy')
    # top_cohorts = 200
    # threshold = 1.7206447124481201
    # eval_frames = 100
    num_eval = 10
    # Evaluation code
    if args.eval is True:
        sc, lab, trials = model.evaluateFromList(
            args.test_list,
            cohorts_path=args.cohorts_path,
            print_interval=100,
            eval_frames=args.eval_frames)
        target_fa = np.linspace(10, 0, num=50)
        result = tuneThresholdfromScore(sc, lab, target_fa)
        print('tfa [thre, fpr, fnr]')
        best_sum_rate = 999
        best_tfa = None
        for i, tfa in enumerate(target_fa):
            print(tfa, result[0][i])
            sum_rate = result[0][i][1] + result[0][i][2]
            if sum_rate < best_sum_rate:
                best_sum_rate = sum_rate
                best_tfa = result[0][i]
        print(f'Best sum rate {best_sum_rate} at {best_tfa}')
        print(f'EER {result[1]} at threshold {result[2]}')
        print(f'AUC {result[3]}')
        sys.exit(1)

    # Test code
    if args.test is True:
        model.testFromList(args.test_path,
                           cohorts_path=args.cohorts_path,
                           thre_score=args.test_threshold,
                           print_interval=100,
                           eval_frames=args.eval_frames)
        sys.exit(1)

    # Prepare embeddings for cohorts/verification
    if args.prepare is True:
        model.prepare(eval_frames=args.eval_frames,
                      from_path=args.test_list,
                      save_path=args.save_path,
                      num_eval=num_eval,
                      prepare_type=args.prepare_type)
        sys.exit(1)

    # Predict
    if args.predict is True:
        """
        Predict new utterance based on distance between its embedding and saved embeddings.
        """
        embeds_path = Path(args.save_path, 'embeds.pt')
        classes_path = Path(args.save_path, 'classes.npy')
        embeds = torch.load(embeds_path).to(torch.device(args.device))
        classes = np.load(str(classes_path), allow_pickle=True).item()
        if args.test_list.endswith('.txt'):
            files = []
            with open(args.test_list) as listfile:
                while True:
                    line = listfile.readline()
                    if not line:
                        break
                    data = line.split()

                    # Append random label if missing TODO: ? why random
                    if len(data) == 2:
                        data = [random.randint(0, 1)] + data

                    files.append(Path(data[1]))
                    files.append(Path(data[2]))

            files = list(set(files))
        else:
            files = list(Path(args.test_list).glob('*/*.wav'))
        files.sort()

        same_smallest_score = 1
        diff_biggest_score = 0
        for f in tqdm(files):
            embed = model.embed_utterance(f,
                                          eval_frames=args.eval_frames,
                                          num_eval=num_eval,
                                          normalize=model.__L__.test_normalize)
            embed = embed.unsqueeze(-1)
            dist = F.pairwise_distance(embed, embeds).detach().cpu().numpy()
            dist = np.mean(dist, axis=0)
            score = 1 - np.min(dist) ** 2 / 2
            if classes[np.argmin(dist)] == f.parent.stem:
                if score < same_smallest_score:
                    same_smallest_score = score
                indexes = np.argsort(dist)[:2]
                if fabs((1 - dist[indexes[0]] ** 2 / 2) - (1 - dist[indexes[1]] ** 2 / 2)) < 0.001:
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx] ** 2 / 2
                        if i == 0:
                            tqdm.write(f'+ {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
                else:
                    tqdm.write(f'+ {f}, {score}', end='')
                    if score < args.test_threshold:
                        tqdm.write(' ***', end='')
                    tqdm.write('')
            else:
                if score > diff_biggest_score:
                    diff_biggest_score = score
                if score > args.test_threshold:
                    indexes = np.argsort(dist)[:3]
                    # TODO: duplicated from line 112 - 119
                    for i, idx in enumerate(indexes):
                        score = 1 - dist[idx] ** 2 / 2
                        if i == 0:
                            tqdm.write(f'- {f}, {score} - {classes[idx]}',
                                       end='; ')
                        else:
                            tqdm.write(f'{score} - {classes[idx]}', end='; ')
                    tqdm.write('***')
        print(f'same_smallest_score: {same_smallest_score}')
        print(f'diff_biggest_score: {diff_biggest_score}')
        sys.exit(1)