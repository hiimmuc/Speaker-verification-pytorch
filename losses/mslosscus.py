import torch
from config import *
from torch import nn
from torch.nn.functional import embedding
from utils import *

# from ret_benchmark.losses.registry import LOSS


# @LOSS.register('ms_loss')
class LossFunction(nn.Module):
    def __init__(self, margin=0.1, scale_neg=50, scale_pos=2, ** kwargs):
        super(LossFunction, self).__init__()
        self.thresh = 1.0
        self.epsilon = 0.1
        self.margin = margin
        self.scale_pos = scale_pos
        self.scale_neg = scale_neg

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        # feat: batch_size x outdim
        emb = F.normalize(feats)
        adjacency = torch.eq(labels, labels.t())
        adjacency_not = torch.logical_not(adjacency)
        mask_pos = adjacency.float() - torch.eye(batch_size).float().cuda()
        mask_neg = adjacency_not.float()

        sim_mat = torch.matmul(feats, torch.t(feats))
        sim_mat = torch.maximum(sim_mat, 0)
        pos_mat = torch.matmul(mask_pos, sim_mat)
        neg_mat = torch.matmul(mask_neg, sim_mat)

        pos_exp = torch.exp(-self.scale_pos * (pos_mat - self.thresh))
        pos_exp = torch.where(mask_pos > 0, pos_exp, torch.zeros_like(pos_exp))
        neg_exp = torch.exp(self.scale_neg * (neg_mat - self.thresh))
        neg_exp = torch.where(mask_neg > 0, neg_exp, torch.zeros_like(neg_exp))

        pos_term = torch.log(1 + torch.sum(pos_exp, dim=1)) / self.scale_pos
        neg_term = torch.log(1 + torch.sum(neg_exp, dim=1)) / self.scale_neg

        loss = torch.mean(pos_term + neg_term)
        prec = 0
        return loss
