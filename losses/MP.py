"""Multi proxies"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy
import numpy as np
from utils import accuracy
from .mpa_utils import *


class MP(torch.nn.Module):
    def __init__(self, nClasses=5994, nOut=512, w_init = 10.0, b_init = -5.0, lambda_init = 0.5, **kwargs): #default margin=0.1 alpha=32
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, nOut).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.criterion  = torch.nn.CrossEntropyLoss()
        
        #self.a = nn.Parameter(torch.tensor(alpha))
        #self.m = nn.Parameter(torch.tensor(mrg))
        
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))
        
        self.w2 = nn.Parameter(torch.tensor(w_init))
        self.b2 = nn.Parameter(torch.tensor(b_init))
        
        self.nb_classes = n_classes
        self.sz_embed = nOut
        self._lambda = lambda_init
        
    def forward(self, X, T):
        
        #print("this is mp")
        
        query, centroid, new_label = pre_process((X,T))
        out_positive = torch.stack(query)
        out_anchor = torch.stack(centroid)

        T = torch.LongTensor(new_label)

        
        P = F.normalize(self.proxies, p=2, dim=1)

        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        T_others = torch.from_numpy(numpy.delete(numpy.arange(self.nb_classes), T.detach().cpu().numpy())).long()
        #new_center = P[T_others]
        
        new_center = torch.zeros(self.nb_classes, self.sz_embed).cuda()
        new_center[T] = out_anchor
        new_center[T_others] = P[T_others]
        
        cos_sim_matrix = F.linear(out_positive, new_center) #(batch_size, num_classes)

        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        loss = -torch.sum(P_one_hot * F.log_softmax(cos_sim_matrix, -1), -1)
        
        cos_sim_matrix2 = F.linear(out_anchor, P[T]) #(batch_size, num_classes)

        cos_sim_matrix2 = cos_sim_matrix2 * self.w + self.b   
        
        label       = torch.from_numpy(numpy.asarray(range(0,T.shape[0]))).cuda()
        
        loss2 = self.criterion(cos_sim_matrix2, label)
        
        prec1 = accuracy(cos_sim_matrix2.detach(), label.detach(), topk=(1,))[0]
        
        return loss.mean() + self._lambda * loss2, prec1