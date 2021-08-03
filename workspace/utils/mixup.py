import numpy as np
import torch
import random


def mixup_data(x, y, alpha=0.2, use_cuda=True):
    """x IS A DICT"""
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
#     batch_size = x.size()[0]
    batch_size = x[next(iter(x))].size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
        
    for key, value in x.items():
        x[key] = lam * value + (1 - lam) * value[index, :]
        
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam