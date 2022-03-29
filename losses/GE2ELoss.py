#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Fast re-implementation of the GE2E loss (https://arxiv.org/abs/1710.10467)
# Numerically checked against https://github.com/cvqluu/GE2E-Loss

# import pdb
# import time

# import numpy
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from utils import accuracy


# class LossFunction(nn.Module):

#     def __init__(self, init_w=10.0, init_b=-5.0, **kwargs):
#         super(LossFunction, self).__init__()

#         self.test_normalize = True

#         self.w = nn.Parameter(torch.tensor(init_w))
#         self.b = nn.Parameter(torch.tensor(init_b))
#         self.criterion = torch.nn.CrossEntropyLoss()

#         print('Initialised GE2E')

#     def forward(self, x, label=None):

#         assert x.size()[1] >= 2

#         device = x.get_device()

#         gsize = x.size()[1]
#         centroids = torch.mean(x, 1)
#         stepsize = x.size()[0]

#         cos_sim_matrix = []

#         for ii in range(0, gsize):
#             idx = [*range(0, gsize)]
#             idx.remove(ii)
#             exc_centroids = torch.mean(x[:, idx, :], 1)
#             cos_sim_diag = F.cosine_similarity(x[:, ii, :], exc_centroids)
#             cos_sim = F.cosine_similarity(
#                 x[:, ii, :].unsqueeze(-1), centroids.unsqueeze(-1).transpose(0, 2))
#             cos_sim[range(0, stepsize), range(0, stepsize)] = cos_sim_diag
#             cos_sim_matrix.append(torch.clamp(cos_sim, 1e-6))

#         cos_sim_matrix = torch.stack(cos_sim_matrix, dim=1)

#         torch.clamp(self.w, 1e-6)
#         cos_sim_matrix = cos_sim_matrix * self.w + self.b

#         label = torch.from_numpy(numpy.asarray(range(0, stepsize))).cuda(device=device)
#         nloss = self.criterion(cos_sim_matrix.view(-1, stepsize),
#                                torch.repeat_interleave(label, repeats=gsize, dim=0).cuda(device=device))
#         prec1 = accuracy(cos_sim_matrix.view(-1, stepsize).detach(),
#                          torch.repeat_interleave(label, repeats=gsize, dim=0).detach(), topk=(1,))[0]

#         return nloss, prec1
   
"""PyTorch implementation of GE2E loss"""
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F


class GE2ELoss(nn.Module):
    """Implementation of the GE2E loss in https://arxiv.org/abs/1710.10467
    Accepts an input of size (N, M, D)
        where N is the number of speakers in the batch,
        M is the number of utterances per speaker,
        and D is the dimensionality of the embedding vector (e.g. d-vector)
    Args:
        - init_w (float): the initial value of w in Equation (5)
        - init_b (float): the initial value of b in Equation (5)
    """

    def __init__(self, init_w=10.0, init_b=-5.0, loss_method="softmax"):
        super(GE2ELoss, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([init_w]))
        self.b = nn.Parameter(torch.FloatTensor([init_b]))
        self.loss_method = loss_method

        assert self.loss_method in ["softmax", "contrast"]

        if self.loss_method == "softmax":
            self.embed_loss = self.embed_loss_softmax
        if self.loss_method == "contrast":
            self.embed_loss = self.embed_loss_contrast

    def cosine_similarity(self, dvecs):
        """Calculate cosine similarity matrix of shape (N, M, N)."""
        n_spkr, n_uttr, d_embd = dvecs.size()

        dvec_expns = dvecs.unsqueeze(-1).expand(n_spkr, n_uttr, d_embd, n_spkr)
        dvec_expns = dvec_expns.transpose(2, 3)

        ctrds = dvecs.mean(dim=1).to(dvecs.device)
        ctrd_expns = ctrds.unsqueeze(0).expand(n_spkr * n_uttr, n_spkr, d_embd)
        ctrd_expns = ctrd_expns.reshape(-1, d_embd)

        dvec_rolls = torch.cat([dvecs[:, 1:, :], dvecs[:, :-1, :]], dim=1)
        dvec_excls = dvec_rolls.unfold(1, n_uttr - 1, 1)
        mean_excls = dvec_excls.mean(dim=-1).reshape(-1, d_embd)

        indices = _indices_to_replace(n_spkr, n_uttr).to(dvecs.device)
        ctrd_excls = ctrd_expns.index_copy(0, indices, mean_excls)
        ctrd_excls = ctrd_excls.view_as(dvec_expns)

        return F.cosine_similarity(dvec_expns, ctrd_excls, 3, 1e-6)

    def embed_loss_softmax(self, dvecs, cos_sim_matrix):
        """Calculate the loss on each embedding by taking softmax."""
        n_spkr, n_uttr, _ = dvecs.size()
        indices = _indices_to_replace(n_spkr, n_uttr).to(dvecs.device)
        losses = -F.log_softmax(cos_sim_matrix, 2)
        return losses.flatten().index_select(0, indices).view(n_spkr, n_uttr)

    def embed_loss_contrast(self, dvecs, cos_sim_matrix):
        """Calculate the loss on each embedding by contrast loss."""
        N, M, _ = dvecs.shape
        L = []
        for j in range(N):
            L_row = []
            for i in range(M):
                centroids_sigmoids = torch.sigmoid(cos_sim_matrix[j, i])
                excl_centroids_sigmoids = torch.cat(
                    (centroids_sigmoids[:j], centroids_sigmoids[j + 1 :])
                )
                L_row.append(
                    1.0
                    - torch.sigmoid(cos_sim_matrix[j, i, j])
                    + torch.max(excl_centroids_sigmoids)
                )
            L_row = torch.stack(L_row)
            L.append(L_row)
        return torch.stack(L)

    def forward(self, dvecs):
        """Calculate the GE2E loss for an input of dimensions (N, M, D)."""
        cos_sim_matrix = self.cosine_similarity(dvecs)
        torch.clamp(self.w, 1e-6)
        cos_sim_matrix = cos_sim_matrix * self.w + self.b
        L = self.embed_loss(dvecs, cos_sim_matrix)
        return L.sum()


@lru_cache(maxsize=5)
def _indices_to_replace(n_spkr, n_uttr):
    indices = [
        (s * n_uttr + u) * n_spkr + s for s in range(n_spkr) for u in range(n_uttr)
    ]
    return torch.LongTensor(indices)