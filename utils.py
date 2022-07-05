import collections
import contextlib
import glob
import os
import random
import sys
import time
import wave
from argparse import Namespace

import numpy as np
from numpy.linalg import norm

import soundfile as sf
from pydub import AudioSegment

import torch
import torch.nn as nn
import torch.nn.functional as F

import webrtcvad
import yaml
from matplotlib import pyplot as plt
from matplotlib import cm, colors

from sklearn import metrics
from sklearn.metrics import precision_recall_curve

from scipy import signal
from scipy import spatial
from scipy.io import wavfile
import scipy.signal as sps

import pdb
from operator import itemgetter


## model utils
def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    # tensor.topk -> tensor, long tensor, return the k largest values along dim
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # Computes element-wise equality
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    # calculate number of true values/ batchsize
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef: float = 0.97):
        super().__init__()
        self.coef = coef
        # make kernel
        # In pytorch, the convolution operation uses cross-correlation. So, filter is flipped.
        self.register_buffer(
            'flipped_filter',
            torch.FloatTensor([-self.coef, 1.]).unsqueeze(0).unsqueeze(0))

    def forward(self, input: torch.tensor) -> torch.tensor:
        assert len(input.size(
        )) == 2, 'The number of dimensions of input tensor must be 2!'
        # reflect padding to match lengths of in/out
        input = input.unsqueeze(1)
        input = F.pad(input, (1, 0), 'reflect')
        
        return F.conv1d(input, self.flipped_filter).squeeze(1)


def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    results = {}
    
    labels = np.nan_to_num(labels)
    scores = np.nan_to_num(scores)
    
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    # G-mean
    gmean = np.sqrt(tpr * (1 - fpr))
    idxG = np.argmax(gmean)
    G_mean_result = [idxG, gmean[idxG], thresholds[idxG]]
    
    # ROC
    fnr = 1 - tpr

    fnr = fnr * 100
    fpr = fpr * 100

    tunedThreshold = []
    if target_fr:
        for tfr in target_fr:
            idx = np.nanargmin(np.absolute((tfr - fnr)))
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    for tfa in target_fa:
        idx = np.nanargmin(np.absolute((tfa - fpr)))
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])

    idxE = np.nanargmin(np.absolute((fnr - fpr)))  # index of min fpr - fnr = fpr + tpr - 1
    eer = np.mean([fpr[idxE], fnr[idxE]])  # EER in % = (fpr + fnr) /2
    optimal_threshold = thresholds[idxE]
    
    # precision recall
    precision, recall, thresholds_ = precision_recall_curve(labels, scores, pos_label=1)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)

    # locate the index of the largest f score
    ixPR = np.argmax(fscore)
    # 
    results['gmean'] = G_mean_result
    results['roc'] = [tunedThreshold, eer, metrics.auc(fpr, tpr), optimal_threshold]
    results['prec_recall'] = [precision, recall, fscore[ixPR], thresholds_[ixPR]]
    return results

# ===================================Similarity===================================
def similarity_measure(method='cosine', ref=None, com=None, **kwargs):
    if method == 'cosine':
        return cosine_similarity(ref, com, **kwargs)
    elif method == 'pnorm':
        return pnorm_similarity(ref, com, **kwargs)
    elif method == 'zt_norm':
        return ZT_norm_similarity(ref, com, **kwargs)
    

def ZT_norm_similarity(ref, com, cohorts, top=-1):
    """
    Adaptive symmetric score normalization using cohorts from eval data
    """

    def ZT_norm(ref, com, top=-1):
        """
        Perform Z-norm or T-norm depending on input order
        """
        S = np.mean(np.inner(cohorts, ref), axis=1)
        S = np.sort(S, axis=0)[::-1][:top]
        mean_S = np.mean(S)
        std_S = np.std(S)
        score = np.inner(ref, com)
        score = np.mean(score)
        return (score - mean_S) / std_S

    def S_norm(ref, com, top=-1):
        """
        Perform S-norm
        """
        return (ZT_norm(ref, com, top=top) + ZT_norm(com, ref, top=top)) / 2

    ref = ref.cpu().numpy()
    com = com.cpu().numpy()
    return S_norm(ref, com, top=top)
    
def cosine_similarity(ref, com, **kwargs):
    return np.mean(abs(F.cosine_similarity(ref, com, dim=-1, eps=1e-05)).cpu().numpy())

def pnorm_similarity(ref, com, p=2, **kwargs):
    pdist = F.pairwise_distance(ref, com, p=p, eps=1e-06, keepdim=True)
    return np.mean(pdist.numpy())

## main.py utils
def read_config(config_path, args=None):
    if args is None:
        args = Namespace()
    with open(config_path, "r") as f:
        yml_config = yaml.load(f, Loader=yaml.FullLoader)
    for k, v in yml_config.items():
        args.__dict__[k] = v
    return args

def read_log_file(log_file):
    with open(log_file, 'r+') as rf:
        data = rf.readline().strip().replace('\n', '').split(',')
        data = [float(d.split(':')[-1]) for d in data]
    return data

def round_down(num, divisor):
    return num - (num % divisor)

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)    
    
# ---------------------------------------

# Creates a list of false-negative rates, a list of false-positive rates
# and a list of decision thresholds that give those error-rates.
def ComputeErrorRates(scores, labels):

      # Sort the scores from smallest to largest, and also get the corresponding
      # indexes of the sorted scores.  We will treat the sorted scores as the
      # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(*sorted(
          [(index, threshold) for index, threshold in enumerate(scores)],
          key=itemgetter(1)))
    sorted_labels = []
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds

# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# plot loss graph along with training process

def plot_graph(data, x_label, y_label, title, save_path, show=True, color='b-', mono=True, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    if mono:
        plt.plot(data, color=color)
    else:
        for dt in data:
            plt.plot(dt)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.savefig(save_path)
    if show:
        plt.show()
    plt.close()

def plot_acc_loss(acc, loss, x_label, y_label, title, save_path, show=True, colors=['b-', 'r-'], figsize=(10, 6)):
    # Make an example plot with two subplots...
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(2,1,1)
    ax1.plot(acc, colors[0])
    ax1.set(xlabel=x_label[0], ylabel=y_label[0], title=title[0])

    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(loss, colors[1])
    ax2.set(xlabel=x_label[1], ylabel=y_label[1], title=title[1])
    
    fig.tight_layout()
    # Save the full figure...
    fig.savefig(save_path)
    if show:
        plt.show()
    plt.close()

    
def plot_embeds(embeds, labels, fig_path='./example.pdf'):
    embeds = np.mean(np.array(embeds), axis = 1)
    
    label_to_number = {label: i for i, label in enumerate(set(labels), 1)}
    labels = np.array([label_to_number[label] for label in labels])
    
    print(embeds.shape, labels.shape)
    
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')

    # Create a sphere
    r = 1
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0.0:pi:512j, 0.0:2.0*pi:512j]
    x = r*sin(phi)*cos(theta)
    y = r*sin(phi)*sin(theta)
    z = r*cos(phi)
    ax.plot_surface(
        x, y, z,  rstride=1, cstride=1, color='w', alpha=0.3, linewidth=0)
    ax.scatter(embeds[:,0], embeds[:,1], embeds[:,2], c=labels, s=20)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_aspect("auto")
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

    
def plot_from_file(result_save_path, show=False):
    '''Plot graph from score file

    Args:
        result_save_path (str): path to model folder
        show (bool, optional): Whether to show the graph. Defaults to False.
    '''
    with open(os.path.join(result_save_path , 'scores.txt')) as f:
        line_data = f.readlines()

    line_data = [line.strip().replace('\n', '').split(',')
                 for line in line_data]

    data = [{}]
    data_val = [{}]
    last_epoch = 1
    step = 10
    for line in line_data:
        if 'IT' in line[0]:
            epoch = int(line[0].split(' ')[-1])

            if epoch not in range(last_epoch - step, last_epoch + 2):
                data.append({})

            data[-1][epoch] = line
            last_epoch = epoch

    for i, dt in enumerate(data):
        data_loss = [float(line[3].strip().split(' ')[1])
                     for _, line in dt.items()]
        data_acc = [float(line[2].strip().split(' ')[1])
                    for _, line in dt.items()]
        plot_acc_loss(acc=data_acc, 
                      loss=data_loss, 
                      x_label=['epoch', 'epoch'], 
                      y_label=['accuracy', 'loss'],
                      title=['Accuracy', 'Loss'],
                      figsize=(10, 12),
                      save_path=f"{result_save_path}/graph.png", show=show)
        plt.close()
        
    # val plot
    if os.path.isfile(f"{result_save_path}/val_log.txt"):
        with open(f"{result_save_path}/val_log.txt") as f:
            val_line_data = f.readlines()

        val_line_data = [line.strip().replace('\n', '').split(',')
                     for line in val_line_data]

        for line in val_line_data:
            if 'IT' in line[0]:
                epoch = int(line[0].split(' ')[-1])

                if epoch not in range(last_epoch - step, last_epoch + step + 1):
                    data_val.append({})

                data_val[-1][epoch] = line
                last_epoch = epoch

        for i, dt in enumerate(data_val):
            data_loss = [float(line[-1].strip().split(' ')[1])
                         for _, line in dt.items()]
            plot_graph(data_loss, 'epoch', 'loss', 'Loss',
                       f"{result_save_path}/val_graph_{i}.png", color='b', mono=True, show=show)
            plt.close()
        
# ---------------------------------------------- linh tinh-------------------------------#
def cprint(text, fg=None, bg=None, style=None, **kwargs):
    """
    Colour-printer.
        cprint( 'Hello!' )                                  # normal
        cprint( 'Hello!', fg='g' )                          # green
        cprint( 'Hello!', fg='r', bg='w', style='bx' )      # bold red blinking on white
    List of colours (for fg and bg):
        k   black
        r   red
        g   green
        y   yellow
        b   blue
        m   magenta
        c   cyan
        w   white
    List of styles:
        b   bold
        i   italic
        u   underline
        s   strikethrough
        x   blinking
        r   reverse
        y   fast blinking
        f   faint
        h   hide
    """

    COLCODE = {
        'k': 0, # black
        'r': 1, # red
        'g': 2, # green
        'y': 3, # yellow
        'b': 4, # blue
        'm': 5, # magenta
        'c': 6, # cyan
        'w': 7  # white
    }

    FMTCODE = {
        'b': 1, # bold
        'f': 2, # faint
        'i': 3, # italic
        'u': 4, # underline
        'x': 5, # blinking
        'y': 6, # fast blinking
        'r': 7, # reverse
        'h': 8, # hide
        's': 9, # strikethrough
    }

    # properties
    props = []
    if isinstance(style,str):
        props = [ FMTCODE[s] for s in style ]
    if isinstance(fg,str):
        props.append( 30 + COLCODE[fg] )
    if isinstance(bg,str):
        props.append( 40 + COLCODE[bg] )

    # display
    props = ';'.join([ str(x) for x in props ])
    if props:
        print(f'\x1b[{props}m' + str(text) + '\x1b[0m', **kwargs)
    else:
        print(text, **kwargs)

if __name__ == '__main__':
    pass
