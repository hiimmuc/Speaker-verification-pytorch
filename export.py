import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from model import SpeakerNet
import argparse

def export_model(args, check=True):
    model = SpeakerNet(**vars(args))
    
    # priority: define weight -> best weight -> last weight
    if args.initial_model_infer:
        chosen_model_state = args.initial_model_infer
    elif os.path.exists(f'{model_save_path}/best_state.pt'):
        chosen_model_state = f'{model_save_path}/best_state.pt'
    else:
        model_files = glob.glob(os.path.join(
            model_save_path, 'model_state_*.model'))
        chosen_model_state = model_files[-1]
    print("Export from ", chosen_model_state)
    
    model.export_onnx(chosen_model_state, check=check)
    
