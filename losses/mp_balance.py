import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy
import numpy as np
from utils import accuracy
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

class MP_Balance(torch.nn.Module):
    def __init__(self, n_classes=5994, sz_embed=512, w_init = 10.0, b_init = -5.0, lambda_init = 0.5): #default margin=0.1 alpha=32
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(n_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
        self.criterion  = torch.nn.CrossEntropyLoss()
        
        #self.a = nn.Parameter(torch.tensor(alpha))
        #self.m = nn.Parameter(torch.tensor(mrg))
        
        self.w = nn.Parameter(torch.tensor(w_init))
        self.b = nn.Parameter(torch.tensor(b_init))
        
        self.w2 = nn.Parameter(torch.tensor(w_init))
        self.b2 = nn.Parameter(torch.tensor(b_init))
        
        self.nb_classes = n_classes
        self.sz_embed = sz_embed
        self._lambda = lambda_init
        
    def forward(self, X, T):
        
        
        #print(X.shape)   (batch size, 2, 512)
        #print("this is mp_balance")
        out_anchor      = torch.mean(X[:,1:,:],1)
        #print(out_anchor.shape)
        out_positive    = X[:,0,:]
        
        P = F.normalize(self.proxies, p=2, dim=1)
        #print("P:",P.shape)
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        T_others = torch.from_numpy(numpy.delete(numpy.arange(self.nb_classes), T.detach().cpu().numpy())).long()
        #new_center = P[T_others]
        #print(T_others)
        new_center = torch.zeros(self.nb_classes, self.sz_embed).cuda()
        new_center[T] = out_anchor
        new_center[T_others] = P[T_others]
        
        cos_sim_matrix = F.linear(out_positive, new_center) #(batch_size, num_classes)

        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        
        loss = -torch.sum(P_one_hot * F.log_softmax(cos_sim_matrix, -1), -1)
        
        cos_sim_matrix2 = F.linear(out_anchor, P[T]) #(batch_size, num_classes)

        cos_sim_matrix2 = cos_sim_matrix2 * self.w + self.b   
        
        label       = torch.from_numpy(numpy.asarray(range(0,X.shape[0]))).cuda()
        
        loss2 = self.criterion(cos_sim_matrix2, label)
        
        #print(self._lambda)
        prec1 = accuracy(cos_sim_matrix2.detach(), label.detach(), topk=(1,))[0]
        
        return loss.mean() + self._lambda * loss2, prec1
