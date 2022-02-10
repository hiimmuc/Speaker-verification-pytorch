import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy
import numpy as np
from utils.utils import accuracy

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

#@jit(nopython=True, parallel=True)
def pre_process(entry):
    
    X,T = entry
    inp = list(set(T))    
    cur_tensor = [torch.stack([X[j] for j in range(len(T)) if(T[j] == inp[i])]) for i in range(len(inp))]
    new_label = [inp[i] for i in range(len(inp)) if(len(cur_tensor) > 1)]
    centroid = [torch.mean(cur_tensor[i][1:],dim=0) for i in range(len(inp)) if(len(cur_tensor) > 1)]
    query = [cur_tensor[i][0] for i in range(len(inp)) if(len(cur_tensor) > 1)]
            
    return query, centroid, new_label


class MP(torch.nn.Module):
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
        
        #print("this is mp")
        
        #print(len(X))
        #print(T.shape)
        query, centroid, new_label = pre_process((X,T))
        #print(len(query))
        #print(len(centroid))
        #print(len(new_label))

        out_positive = torch.stack(query)
        out_anchor = torch.stack(centroid)
        #print("out_positive:",out_positive.shape)
        #print("out_anchor:",out_anchor.shape)
        T = torch.LongTensor(new_label)
        #print(T.shape)
        
        P = F.normalize(self.proxies, p=2, dim=1)
        #P = self.proxies
        P_one_hot = binarize(T = T, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
        
        T_others = torch.from_numpy(numpy.delete(numpy.arange(self.nb_classes), T.detach().cpu().numpy())).long()
        
        new_center = torch.zeros(self.nb_classes, self.sz_embed).cuda()
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
        
        label       = torch.from_numpy(numpy.asarray(range(0,X.shape[0]))).long().cuda()
        #label =  torch.from_numpy(numpy.asarray(range(0,X.shape[0]))).long()
        
        loss_regulator = self.criterion(cos_sim_matrix2, label).mean()
        
        loss = loss_mmp + self._lambda * loss_regulator
        
        prec1 = accuracy(cos_sim_matrix2.detach(), label.detach(), topk=(1,))[0]
        #print(self._lambda)
        
        return loss.mean() + self._lambda * loss, prec1