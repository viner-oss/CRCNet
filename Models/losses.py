import numpy as np
import torch
import torch as th
from torch import nn

def cross_entropy_loss(proportions, device):
    counts = torch.tensor(proportions, dtype=torch.float32)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum()
    
    return nn.CrossEntropyLoss(weight=weights).to(device)

import torch
import torch.nn as nn

def jointloss(proportions, coef1, coef2, device):
    return cross_entropy_loss(proportions, device), nn.MSELoss().to(device)



    
