
import csv
# import time
from pathlib import Path

# import librosa
# import numpy as np
# # import pandas as pd
# import soundfile as sf
# import torch
# import torch.nn.functional as F
# from numpy.core.fromnumeric import argmin
from torchsummary import summary
from tqdm.auto import tqdm
# from torchviz import make_dot
# from utils import *
# from argparse import Namespace

# from models.CoAtNet.CoAtNet import *
from models.Raw_ECAPA import *



def overwirte():
    """fix same name"""
    test = Path("exp/dump/submission_backbone_softmax_normthres.csv")  # 86.55%

    ref = Path("exp/dump/submission_model_pretrain_resnet34v2_rawcode.csv")
    lines = []
    scores = []
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            lines.append([row[0], row[1]])
    with open(test, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in spamreader:
            scores.append([row[2], row[3]])
    new_lines = [lines[i] + scores[i] for i in range(len(lines))]
    with open(test, 'w', newline='') as wf:
        spamwriter = csv.writer(wf, delimiter=',')
        spamwriter.writerow(['audio_1', 'audio_2', 'label', 'score'])
        for line in tqdm(new_lines):
            spamwriter.writerow(line)
    print('Done!')
    pass


def check_result(answer):
    ref = Path("exp\data_truth.txt")
    com = Path(answer)

    reference = {}
    comparation = {}
    #  read from reference csv file
    with open(ref, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=' ')
        # next(spamreader, None)
        for row in tqdm(spamreader, desc="read reference"):
            # print(row[1:])
            key = f"{row[1]}/{row[2]}"
            reference[key] = row[0]

    #  read from compare csv file
    with open(com, newline='') as rf:
        spamreader = csv.reader(rf, delimiter=',')
        next(spamreader, None)
        for row in tqdm(spamreader, desc="read compare"):
            key = f"{row[0]}/{row[1]}"
            comparation[key] = row[2]

    assert len(reference) == len(comparation) or len(reference) == len(comparation) + 1, "length of reference and comparation is not equal"

    # precision
    count = 0
    for k, v in comparation.items():
        if reference[k] == v:
            count += 1
        pass
    print(count/len(comparation), '%')


if __name__ == '__main__':
    # check_inequality_data()
    # create_full_augmented_dataset()
    # check_result("exp\dump\submission_list_test_1310_ambase.csv")

    model = MainModel(n_mels=80, max_frames=100, sample_rate=8000, augment=True)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("nb_params:{}".format(nb_params))
    # audio =
    model.to('cpu')
    summary(model, (8120, ), batch_size=1, device='cpu')
#     dummy_input = torch.randnt((16240, ))
#     make_dot(yhat, params=dict(list(model.named_parameters()))).render("rnn_torchviz", format="png")
#     print(model)
#     from audio_utils import *
    
#     print(get_audio_ffmpeg_astats(r'Speaker-verification-pytorch-master\dataset\check-log\20220208_101221_ref.wav'))
#     pass