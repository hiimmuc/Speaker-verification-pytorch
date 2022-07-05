import csv
import glob
import os
import random
import sys
import subprocess

import time
from math import fabs
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.metrics import fbeta_score, accuracy_score

from model import SpeakerNet
from utils import tuneThresholdfromScore, read_config
from inference import *

def bendmark_models(model_dir, eval_config_file, test_config_file):
    model_paths = glob.glob(model_dir + '/*.pt') + glob.glob(model_dir + '/*.model')
    scoring_mode = 'cosine'
    num_eval = 20
    
    for model_path in tqdm(model_paths):
        chosen_model_state = model_path
        
        args = read_config(eval_config_file)
        args.initial_model_infer = chosen_model_state
        
        model = SpeakerNet(**vars(args))
        # path here
        model_save_path = os.path.join(args.save_path , 
                                       f"{args.model}/model")
        result_save_path = os.path.join(args.save_path , 
                                        f"{args.model}/result")
        # Write args to score_file
        settings_file = open(result_save_path + '/settings.txt', 'a+')
        score_file = open(result_save_path + "/Inference_log.txt", "a+")
        test_log_file = open(result_save_path + "/Testing_log.txt", "a+")
        print(f'Loading model from {chosen_model_state}')
        
        model.loadParameters(chosen_model_state)
        model.eval()
        #### eval model to have the threshold
        sc, lab, trials = model.evaluateFromList(
            args.test_list,
            cohorts_path=args.cohorts_path,
            print_interval=1,
            eval_frames=args.eval_frames,
            scoring_mode=scoring_mode)
        target_fa = np.linspace(5, 0, num=50)
        result = tuneThresholdfromScore(sc, lab, target_fa) 
        threshold = result['gmean'][2]
        # write to file
        write_file = Path(result_save_path, 'evaluation_results.txt')
        
        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'label', 'predict_label','score'])
            preds = []
            for score, label, pair in zip(sc, lab, trials):
                pred = int(score >= result['gmean'][2])
                com, ref = pair.strip().split(' ')
                spamwriter.writerow([com, ref, label, pred, score])
                preds.append(pred)
            
            # print out metrics results
            beta_values=[0.5, 2]
            prec_recall = evaluate_by_precision_recall(lab, preds, beta_values=beta_values)
            print("REPORT:\n", prec_recall[0])
            print("Accuracy for each class:", f"\n0's: {prec_recall[1][0]}\n1's: {prec_recall[1][1]}")
            for b in beta_values:
                print(f"F-{b}:", prec_recall[2][b])
        
        print('+++++++++++++++++++++++++++++++++++++++++++\n')
        ### Testmodel
        args = read_config(test_config_file)
        args.initial_model_infer = chosen_model_state
        model.eval()
        model.testFromList(args.test_path,
                           args.test_list,
                           cohorts_path=args.cohorts_path,
                           thre_score=threshold,
                           print_interval=1,
                           eval_frames=args.eval_frames,
                           scoring_mode=scoring_mode,
                           output_file=args.com, num_eval = num_eval)
        
        roc, prec_recall = evaluate_result(path=args.com, ref=args.ref)
        test_log_file.writelines([f">{time.strftime('%Y-%m-%d %H:%M:%S')}<",
                                  f"Test result on: [{args.test_list}] with [{args.initial_model_infer}]\n",
                                  f"Threshold: {threshold}\n",
                                  f"ROC: {roc}\n",
                                  f"Report: \n{prec_recall}\n",
                                  f"Save to {args.com} and {args.ref} \n========================================\n"])
        test_log_file.close()
    sys.exit(1)
    
        
if __name__ == '__main__':
    model_path = "backup/1001/Raw_ECAPA/ARmSoftmax/model_backup"
    eval_path = ""
    bendmark_models(model_path, "backup/config/config_eval.yaml",  "backup/config/config_test.yaml")
        

        