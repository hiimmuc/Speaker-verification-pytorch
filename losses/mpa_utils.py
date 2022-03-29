import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy
import numpy as np
from utils import accuracy

# from numba import jit, prange
# from fastai.core import parallel
#from pytorch_metric_learning import miners, losses

def binarize(T, nb_classes):
    T = T.cpu().numpy()
    import sklearn.preprocessing
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T


def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

def pre_process(entry):
    
    X,T = entry
    inp = list(set(T))    
    cur_tensor = [torch.stack([X[j] for j in range(len(T)) if(T[j] == inp[i])]) for i in range(len(inp))]
    new_label = [inp[i] for i in range(len(inp)) if(len(cur_tensor) > 1)]
    centroid = [torch.mean(cur_tensor[i][1:],dim=0) for i in range(len(inp)) if(len(cur_tensor) > 1)]
    query = [cur_tensor[i][0] for i in range(len(inp)) if(len(cur_tensor) > 1)]
            
    return query, centroid, new_label