import torch
import numpy as np

def cosine_annealinglr_v1(optimizer, test_interval, lr_decay, T_max=10000, lr_min=1e-6, lr=1e-3,  **kwargs):
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            T_max,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / lr,
        ),
    )
    return lr_scheduler, 'epoch'


def cosine_annealinglr(optimizer, T_max, eta_min=0, last_epoch=- 1, verbose=False, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
    return lr_scheduler, 'epoch'


def cosine_annealing_warm_restarts(optimizer, T_max=10000, T_mult=1, eta_min=0, last_epoch=- 1, verbose=False, **kwargs):
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=T_max, T_mult=T_mult, eta_min=eta_min, last_epoch=last_epoch, verbose=verbose)
    return lr_scheduler, 'iteration'


def cycliclr(optimizer, base_lr=1e-8, max_lr=1e-3, T_max=10000, 
              mode='triangular2', gamma=1.0, scale_fn=None, scale_mode='cycle', cycle_momentum=False,  
              base_momentum=0.8, max_momentum=0.9, last_epoch=- 1, verbose=False, **kwargs):

    """
    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """
    step_size_up = T_max // 2
    step_size_down = T_max // 2
    
    lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer,
        base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up, step_size_down=step_size_down, 
        mode=mode, gamma=gamma, scale_fn=scale_fn, scale_mode=scale_mode, cycle_momentum=cycle_momentum, 
        base_momentum=base_momentum, max_momentum=max_momentum, last_epoch=last_epoch, verbose=verbose
    )
    return lr_scheduler, 'iteration'

def steplr(optimizer, step_size, lr_decay, **kwargs):
    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=step_size,
                                              gamma=lr_decay)
    lr_step = 'epoch'
    # print('Initialised step LR scheduler')
    return sche_fn, lr_step


