import torch

# taken from https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/


def Scheduler(optimizer, step_size, lr_decay, **kwargs):
    sche_fn = torch.optim.lr_scheduler.StepLR(optimizer,
                                              step_size=step_size,
                                              gamma=lr_decay)
    lr_step = 'epoch'
    # print('Initialised step LR scheduler')
    return sche_fn, lr_step


