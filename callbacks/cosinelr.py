import torch
import numpy as np

def Scheduler(optimizer, test_interval, lr_decay, lr_min=1e-6, lr=1e-3,  **kwargs):
    def cosine_annealing(step, total_steps, lr_max, lr_min):
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))
    total_steps = 500 * 200 # max epochs * max steps per epoch
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            total_steps,
            1,  # since lr_lambda computes multiplicative factor
            lr_min / lr,
        ),
    )
    return lr_scheduler, 'epoch'