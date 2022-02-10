import csv
import glob
import os
import random
import sys

import time
from math import fabs
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from model import SpeakerNet
from utils.utils import tuneThresholdfromScore


def inference(args):
    model = SpeakerNet(**vars(args))
    model_save_path = args.save_path + f"/{args.model}/model"
    result_save_path = args.save_path + f"/{args.model}/result"
    # Write args to score_file
    settings_file = open(result_save_path + '/settings.txt', 'a+')
    score_file = open(result_save_path + "/Inference_log.txt", "a+")
    # summary settings
    settings_file.write(
        f'\n[INFER]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    score_file.write(
        f'\n[INFER]------------------{time.strftime("%Y-%m-%d %H:%M:%S")}------------------\n')
    # write the settings to settings file
    for items in vars(args):
        # print(items, vars(args)[items])
        settings_file.write('%s %s\n' % (items, vars(args)[items]))
    settings_file.flush()
    settings_file.close()

    # priority: define weight -> best weight -> last weight
    if args.initial_model_infer:
        chosen_model_state = args.initial_model_infer
    elif os.path.exists(f'{model_save_path}/best_state.model'):
        chosen_model_state = f'{model_save_path}/best_state.model'
    else:
        model_files = glob.glob(os.path.join(
            model_save_path, 'model_state_*.model'))
        chosen_model_state = model_files[-1]

    print(f'Loading model from {chosen_model_state}')
    model.loadParameters(chosen_model_state)

    model.eval()
    # set defalut threshold
    threshold = args.test_threshold
    scoring_mode = args.scoring_mode
    num_eval = 10

    # Evaluation code
    if args.eval is True:
        sc, lab, trials = model.evaluateFromList(
            args.test_list,
            cohorts_path=args.cohorts_path,
            print_interval=1,
            eval_frames=args.eval_frames,
            scoring_mode=scoring_mode)
        
        target_fa = np.linspace(5, 0, num=50)
        result = tuneThresholdfromScore(sc, lab, target_fa)
        
#         result form : (tunedThreshold, eer, optimal_threshold, metrics.auc(fpr, tpr), G_mean_result)
#         print('tfa [thre, fpr, fnr]')
        best_sum_rate = 999
        best_tfa = None
        for i, tfa in enumerate(target_fa):
#             print(tfa, result[0][i])
            sum_rate = result[0][i][1] + result[0][i][2]
            if sum_rate < best_sum_rate:
                best_sum_rate = sum_rate
                best_tfa = result[0][i]
        
        print("\n[RESULTS]\n",
              f"Best sum rate {best_sum_rate} at {best_tfa}, AUC {result[3]}\n",
              f">> EER {result[1]}% at threshold {result[2]}\n",
              f">> Gmean result: \n>>> EER: {(1 - result[-1][1]) * 100}% at threshold {result[-1][2]}\n>>> ACC: {result[-1][1] * 100}%")
        
        score_file.writelines(
            [f"Evaluation result on: [{args.test_list}] with [{args.initial_model_infer}]\n",
             f"Best sum rate {best_sum_rate} at {best_tfa}\n",
             f"EER {result[1]} at threshold {result[2]}\nAUC {result[3]}\n",
             f"Gmean result:\n",
             f"EER: {(1 - result[-1][1]) * 100}% at threshold {result[-1][2]}\n ACC: {result[-1][1] * 100}%\n=================>\n"])
        score_file.close()
        
        # write to file
        save_root = args.save_path + f"/{args.model}/result"
        write_file = Path(save_root, 'evaluation_results.txt')
        
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2','score', 'label', 'predict_label'])
            for score, label, pair in zip(sc, lab, trials):
                pred = '1' if score >= result[2] else '0'
                com, ref = pair.strip().split(' ')
                spamwriter.writerow([com, ref, score, label, pred])
            print("=============END===============//\n")

        sys.exit(1)

    # Test list
    if args.test is True:
        model.testFromList(args.test_path,
                           args.test_list,
                           cohorts_path=args.cohorts_path,
                           thre_score=threshold,
                           print_interval=1,
                           eval_frames=args.eval_frames,
                           scoring_mode=scoring_mode,
                           output_file=args.com)
        prec = check_result(path=args.com, ref=args.ref)
        score_file.writelines([f"Test result on: [{args.test_list}] with [{args.initial_model_infer}]\n",
                               f"Threshold: {threshold}\n",
                               f"Precision: {prec}\n",            
                               f"Save to {args.com} and {args.ref} \n=================>\n"])
        score_file.close()
        sys.exit(1)

    # Test pair by pair
    if args.test_by_pair is True:
        model.test_each_pair(args.test_path,
                             cohorts_path=args.cohorts_path,
                             thre_score=threshold,
                             print_interval=1,
                             eval_frames=args.eval_frames,
                             scoring_mode=scoring_mode)
        sys.exit(1)

    # Prepare embeddings for cohorts/verification
    if args.prepare is True:
        model.prepare(eval_frames=args.eval_frames,
                      from_path=args.test_list,
                      save_path=args.cohorts_path,
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

        
def check_result(path="backup/Raw_ECAPA/result/private_test_results.txt", ref="dataset/test_callbot/valid_speaker/private_test_cb_truth.txt"):
    com = path
    
    assert os.path.isfile(ref) and os.path.isfile(com), "Files not exists"
    
    ref_data = {}
    com_data = {}
    
    print("Checking results....")
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=' ')
        for row in spamreader:
            key = f"{row[1]}/{row[-1]}"
            ref_data[key] = str(row[0])
            
    with open(com, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            key = f"{row[0]}/{row[1]}"
            com_data[key] = str(row[2])
            
    assert len(ref_data)==len(com_data), "The length of 2 files is not equal"
    
    count_true = 0
    
    tta = 0
    tfr = 0
    
    ta = 0
    fr = 0
    
    for k, v in com_data.items():
        if ref_data[k] == '1':
            tta += 1
        else:
            tfr += 1
        
        if v == '1':
            ta += 1
        else:
            fr += 1
        
        if ref_data[k] == v:
            count_true += 1
    
    precision = count_true * 100 / len(ref_data)
    print(">> Precision:", precision, 'Error rate:', 100 - precision, 
          f"\n>>> True Accepted:  {ta} pairs / Total '1' label: {tta} pairs",
          f"\n>>> False Rejected:  {fr} pairs / Total '0' label: {tfr} pairs",
          f"\n>>> True:  {count_true} pairs / Total: {len(ref_data)} pairs\n========//")
    return precision
    
if __name__ == '__main__':
    pass