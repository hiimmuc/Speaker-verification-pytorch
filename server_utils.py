import base64
import os
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from pydub import AudioSegment

from model import SpeakerNet
from utils import cosine_simialrity
from processing.wav_conversion import (normalize_audio_amp, np_to_segment, segment_to_np)
from processing.augment import gain_target_amplitude


def normalize_score(score, threshold, fixed_threshold=0.5):
    '''How magic works'''
    ratio = threshold / fixed_threshold
    sign = -1 if ratio <= 1 else 1
    
    if score > threshold:
        score_norm = score * (ratio ** sign)
    else:
        score_norm = score / (ratio ** sign)
        
    result =  score_norm if score_norm <= 1 else 1 # limit score
    return result
    

def decode_audio(audio_data, sr, dtype_np=np.int16):
    audio_data_bytes = audio_data.encode('utf-8') # string to bytes
    audio_data_b64 = base64.decodebytes(audio_data_bytes) # bytes to b64
    audio_data_np = np.frombuffer(audio_data_b64, dtype=dtype_np) #b 64 to np
    return audio_data_np

def preprocess_audio(audio_data_np, target_volume=-10):
    # np to segment -> norm volume -> convert sang np -> norm biên độ
    audio_data_seg = np_to_segment(audio_data_np)
    audio_data_seg = gain_target_amplitude(audio_data_seg, target_dBFS=target_volume)
    audio_data_np_new = segment_to_np(audio_data_seg)
    audio_out = normalize_audio_amp(audio_data_np_new)
    return audio_out

def compute_score_by_mean_ref(model, ref_source, com_source, 
                              threshold=0.5, fixed_threshold=0.5,
                              eval_frames=100, num_eval=20,normalize=True, sr=8000,  **kwargs):
    """
    Predict new utterance based on distance between samples of com and ref.
    """   
    mean_scores = []
    ref_emb = model.prepare(source=ref_source, save_path=None, prepare_type='embed', num_eval=num_eval, eval_frames=eval_frames).detach().cpu().numpy()
    
    for data_np_com in com_source:
        com_emb = model.embed_utterance(data_np_com, num_eval=num_eval, eval_frames=eval_frames, normalize=normalize).detach().cpu().numpy()
        
        # dist = F.pairwise_distance(com_emb, ref_emb).detach().cpu().numpy()
        # mean_dist = np.mean(dist, axis=0)
        # mean_score = 1 - np.min(dist) ** 2 / 2

        # indexes = np.argsort(dist)[:top]
        # for idx in indexes:
        #     score = 1 - dist[idx] ** 2 / 2
        #     top_scores.append(score)
        mean_scores.append(cosine_simialrity(torch.from_numpy(ref_emb), torch.from_numpy(com_emb)))
    
    norm_mean_scores = [normalize_score(mean_score, threshold=threshold, fixed_threshold=fixed_threshold) for mean_score in mean_scores]    
    
    return norm_mean_scores, mean_scores

def score_of_pair(ref_emb, com_emb, threshold=0.5, fixed_threshold=0.5):
    score = cosine_simialrity(torch.from_numpy(ref_emb), torch.from_numpy(com_emb))
    score_norm = normalize_score(score, 
                                 threshold=threshold, 
                                 fixed_threshold=fixed_threshold)
    return score_norm, score

def compute_score_by_pair(model, com_source, ref_source, 
                          threshold=0.5, fixed_threshold=0.5,
                          eval_frames=100, num_eval=20,normalize=True, sr=8000,  **kwargs):
    '''
    return max score of all pair test
    '''
    # print("Getting embedings...")
    embedding_ref = []
    for audio_data_np in ref_source:
        # get embedding
        # t0 = time.time()
        emb = np.asarray(model.embed_utterance(audio_data_np, 
                                               eval_frames=eval_frames, 
                                               num_eval=num_eval, 
                                               normalize=normalize, sr=sr))
        # print("Embedding time:", time.time() - t0)
        embedding_ref.append(emb)

    embedding_com = []
    for audio_data_np in com_source:
        # get embedding
        # t0 = time.time()
        emb = np.asarray(model.embed_utterance(audio_data_np, 
                                               eval_frames=eval_frames, 
                                               num_eval=num_eval, 
                                               normalize=normalize, sr=sr))
        # print("Embedding time:", time.time() - t0)
        embedding_com.append(emb)

    # compare embeding
    confidence_scores = []
    nom_confidence_scores = []
    for i, ref_emb in enumerate(embedding_ref):
        confidence_scores.append([])
        nom_confidence_scores.append([])
        for j, com_emb in enumerate(embedding_com):
            norm_confidence_score, confidence_score = score_of_pair(ref_emb, com_emb, 
                                                                    threshold, 
                                                                    fixed_threshold=fixed_threshold)
            confidence_scores[i].append(confidence_score)
            nom_confidence_scores[i].append(norm_confidence_score)
    
    return nom_confidence_scores, confidence_scores


if __name__ == '__main__':
    pass