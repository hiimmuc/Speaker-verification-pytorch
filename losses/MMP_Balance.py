import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy
import numpy as np
from utils import accuracy
from .mpa_utils import *
#from pytorch_metric_learning import miners, losses


class MMP_Balance(torch.nn.Module):
    def __init__(self, nOut=512, nClasses=5994, w_init = 10.0, b_init = -5.0, lambda_init = 0.5, **kwargs): 
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        
        self.proxies = torch.nn.Parameter(torch.randn(nClasses, nOut)).cuda()
        #  self.proxies = torch.nn.Parameter(torch.randn(n_classes, sz_embed))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.criterion  = torch.nn.CrossEntropyLoss()
        
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))
        
        self.nb_classes = nClasses
        self.sz_embed = nOut
        self._lambda = lambda_init
        
    def forward(self, X, T):
        
        #print("this is MMP_Balance")
        #input size(batchsize, 2, feature size)
        assert X.size()[1] >= 2
        
        device = X.get_device()
        
        out_anchor      = torch.mean(X[:,1:,:],1)
        out_positive    = X[:,0,:]
        
        P = F.normalize(self.proxies, p=2, dim=1)
        #P = self.proxies
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes).to(device)
        N_one_hot = 1 - P_one_hot
        
        T_others = torch.from_numpy(numpy.delete(numpy.arange(self.nb_classes), T.detach().cpu().numpy())).long()
        
        new_center = torch.zeros(self.nb_classes, self.sz_embed).to(device)
        #new_center = torch.zeros(self.nb_classes, self.sz_embed)
        
        new_center[T] = l2_norm(out_anchor)
        new_center[T_others] = P[T_others]
        
        l1 = torch.log(torch.exp(-torch.sum(out_positive * new_center[T], dim=1) * self.w + self.b).sum() + 1) #positive pair
        l2 = torch.log(torch.sum((torch.exp(F.linear(out_positive, new_center[T_others])* self.w - self.b)), dim=1) + 1).mean()
        z = torch.exp(F.linear(out_positive,new_center[T])* self.w - self.b)
        l3 = torch.log(torch.sum(z, dim=1) - torch.diag(z) + 1).mean()
        loss_mmp = l1 + l2 + l3

        #print(l2)
        
        cos_sim_matrix2 = F.linear(out_anchor, P[T]) #(batch_size, num_classes)
        cos_sim_matrix2 = cos_sim_matrix2 * self.w - self.b   
        
        label       = torch.from_numpy(numpy.asarray(range(0,X.shape[0]))).long().to(device)
        #label =  torch.from_numpy(numpy.asarray(range(0,X.shape[0]))).long()
        
        loss_regulator = self.criterion(cos_sim_matrix2, label).mean()
        
        loss = loss_mmp + self._lambda * loss_regulator
        
        prec1 = accuracy(cos_sim_matrix2.detach(), label.detach(), topk=(1,))[0]
        
        return loss.mean(), prec1