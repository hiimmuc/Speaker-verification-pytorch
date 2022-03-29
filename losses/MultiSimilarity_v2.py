import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from .pytorch_metric_learning import miners, losses

class MultiSimilarity_v2(torch.nn.Module):
    def __init__(self, **kwargs):
        super(MultiSimilarity_v2, self).__init__()
        self.thresh = 0.5
        self.epsilon = 0.1
        self.scale_pos = 2
        self.scale_neg = 50
        
        self.test_normalize = True
        
        self.miner = miners.MultiSimilarityMiner(epsilon=self.epsilon)
        self.loss_func = losses.MultiSimilarityLoss(self.scale_pos, self.scale_neg, self.thresh)
        
    def forward(self, embeddings, labels):
        assert embeddings.size(0) == labels.size(0), \
            f"feats.size(0): {embeddings.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        # feats: batch x n x d
        ## batchxn -> batch.n n: nPerSpeaker ; size [64] -> [640], repeat by number of utterances
        labels = labels.repeat_interleave(embeddings.size()[1])
        ## batchxnxd -> batch.n x d; [64, 5, 128] -> [320, 128]
        embeddings = embeddings.reshape(-1, embeddings.size()[-1])
        ## feat: batch_size x outdim
        batch_size = embeddings.size(0)
        
        # origin code:        
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        # get accuracy:
        # prec = get_embedding_acc(embeddings, labels)
        ##
        return loss, 0