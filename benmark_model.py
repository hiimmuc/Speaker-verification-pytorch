import csv
import glob
import os
import random
import subprocess
import sys
import time
from math import fabs
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, fbeta_score, roc_curve)
from tqdm import tqdm

from inference import *

import argparse
from model import SpeakerEncoder, WrappedModel, ModelHandling
from utils import read_config, tuneThresholdfromScore


def bendmark_models(model_dir, verification_config_file):
    model_paths = glob.glob(model_dir + '/*.pt') + \
        glob.glob(model_dir + '/*.model')
    scoring_mode = 'cosine'
    num_eval = 20
    
    args = read_config(verification_config_file)
    args = argparse.Namespace(**args)

    for model_path in tqdm(model_paths):
        chosen_model_state = model_path
        
        args.initial_model_infer = chosen_model_state

        net = WrappedModel(SpeakerEncoder(**vars(args)))
        max_iter_size = args.step_size
    
        model = ModelHandling(net, **dict(vars(args), T_max=max_iter_size))
        
        model_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/model")
        result_save_path = os.path.join(
        args.save_folder, f"{args.model['name']}/{args.criterion['name']}/result")
        
        # Write args to score_file
        settings_file = open(result_save_path + '/settings.txt', 'a+')
        score_file = open(result_save_path + "/Inference_log.txt", "a+")
        test_log_file = open(result_save_path + "/Testing_log.txt", "a+")
        print(f'Loading model from {chosen_model_state}')

        model.loadParameters(chosen_model_state)
        model.__model__.eval()
        
        # eval model to have the threshold
        sc, lab, trials = model.evaluateFromList(
            listfilename=args.evaluation_file,
            distributed=False,
            dataloader_options=args.dataloader_options,
            cohorts_path=args.cohorts_path,
            num_eval=args.num_eval,
            scoring_mode=scoring_mode)

        target_fa = np.linspace(5, 0, num=50)

        result = tuneThresholdfromScore(sc, lab, target_fa)
        ####

        best_sum_rate = 999
        best_tfa = None
        for i, tfa in enumerate(target_fa):
            sum_rate = result['roc'][0][i][1] + result['roc'][0][i][2]
            if sum_rate < best_sum_rate:
                best_sum_rate = sum_rate
                best_tfa = result['roc'][0][i]

        print("\n[RESULTS]\nROC:",
              f"Best sum rate {best_sum_rate} at {best_tfa}, AUC {result['roc'][2]}\n",
              f">> EER {result['roc'][1]}% at threshold {result['roc'][-1]}\n",
              f">> Gmean result: \n>>> EER: {(1 - result['gmean'][1]) * 100}% at threshold {result['gmean'][2]}\n>>> ACC: {result['gmean'][1] * 100}%\n",
              f">> F-score {result['prec_recall'][2]}% at threshold {result['prec_recall'][-1]}\n")
        threshold_set =  result['gmean'][-1]
        # write to file
        write_file = Path(result_save_path, 'evaluation_results.txt')

        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(
                ['audio_1', 'audio_2', 'label', 'predict_label', 'score'])
            preds = []
            for score, label, pair in zip(sc, lab, trials):
                pred = int(score >= result['gmean'][2])
                com, ref = pair.strip().split(' ')
                spamwriter.writerow([com, ref, label, pred, score])
                preds.append(pred)

            # print out metrics results
            beta_values = [0.5, 2]
            prec_recall = evaluate_by_precision_recall(
                lab, preds, beta_values=beta_values)
            print("REPORT:\n", prec_recall[0])
            print("Accuracy for each class:",
                  f"\n0's: {prec_recall[1][0]}\n1's: {prec_recall[1][1]}")
            for b in beta_values:
                print(f"F-{b}:", prec_recall[2][b])

        # Testmodel
 
        model.testFromList(args.verification_file,
                           thresh_score=threshold_set,
                           output_file=args.log_test_files['com'],
                           distributed=False,
                           dataloader_options=args.dataloader_options,
                           cohorts_path=args.cohorts_path,
                           num_eval=args.num_eval,
                           scoring_mode=scoring_mode)       
        
        roc, prec_recall = evaluate_result(path=args.log_test_files['com'], ref=args.log_test_files['ref'])
        test_log_file.writelines([f">{time.strftime('%Y-%m-%d %H:%M:%S')}<",
                                  f"Test result on: [{args.verification_file}] with [{args.initial_model_infer}]\n",
                                  f"Threshold: {threshold_set}\n",
                                  f"ROC: {roc}\n",
                                  f"Report: \n{prec_recall}\n",
                                  f"Save to {args.log_test_files['com']} and {args.log_test_files['ref']} \n========================================\n"])
        test_log_file.close()
    sys.exit(1)


if __name__ == '__main__':
    model_path = "backup/3105/save/Raw_ECAPA/ARmSoftmax/model_old"
    bendmark_models(model_path, "yaml/verification.yaml")
