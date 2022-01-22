#benchmark dataset:
import itertools
import os
import time
from argparse import Namespace
from pathlib import Path
import numpy as np
from pydub import AudioSegment
from model import SpeakerNet
from tqdm import tqdm
from utils import *

def all_pairs(lst):
    return list(itertools.combinations(lst, 2))

def check_matching(ref_emb, com_emb, threshold=0.5):
    score = cosine_simialrity(ref_emb, com_emb)
    ratio = threshold / 0.5
    result = (score / ratio) if (score / ratio) < 1 else 1
    matching = result > 0.5
#     print("Result:", result, "Score:", score)
    return matching, result

# load model
threshold = 0.2528385519981384
model_path = str(Path('backup/Raw_ECAPA/model/best_state-278e-2.model'))
config_path = str(Path('backup/Raw_ECAPA/config_deploy.yaml'))
args = read_config(config_path)

t0 = time.time()
model = SpeakerNet(**vars(args))
model.loadParameters(model_path, show_error=False)
model.eval()
print("Model Loaded time: ", time.time() - t0)


# ===================================================
root = 'dataset/train/'
folders = glob.glob(str(Path(root, '*')))

for folder in tqdm(folders[:]):
    files = glob.glob(f"{folder}/*.wav")
    pairs = all_pairs(files)
    files_emb_dict = {}
    imposters = {}
    
    for fn in files:
        emb = model.embed_utterance(fn, eval_frames=100, num_eval=10, normalize=True)
        if fn not in files_emb_dict:
            files_emb_dict[fn] = emb
    for pair in pairs:
        match, score = check_matching(files_emb_dict[pair[0]], files_emb_dict[pair[1]], threshold)
        if not match:
            if pair[0] not in imposters:
                imposters[pair[0]] = 0
            if pair[1] not in imposters:
                imposters[pair[1]] = 0
            imposters[pair[0]] += 1
            imposters[pair[1]] += 1
    imposters_list = [k for k,v in imposters.items() if v >= int(0.2 * len(files))]

    with open("Imposter.txt", 'a+') as f:
        if len(imposters_list) > 0:
            f.write(f"Folder:{folder}\n")
            for imp in sorted(imposters_list):
                f.write(f"[{imposters[imp]}/{len(files)}] - {imp}\n")
            f.write("//================//\n")   