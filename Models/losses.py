import numpy as np
import torch
import torch as th
from torch import nn

def cross_entropy_loss(proportions, device):
    counts = torch.tensor(proportions, dtype=torch.float32)
    inv_freq = 1.0 / counts
    weights = inv_freq / inv_freq.sum()
    return nn.CrossEntropyLoss(weight=weights).to(device)

class JointLoss(nn.Module):
    """
    JointLoss: α * CrossEntropyLoss + β * MSELoss
    """
    def __init__(self, proportions, coef1, coef2, device):
        super().__init__()
        self.coef1 = coef1
        self.coef2 = coef2
        self.mse = nn.MSELoss()
        self.ce = cross_entropy_loss(proportions, device)

    def forward(self, logit, tgt, pred, real):
        assert len(logit.shape) == 2
        assert pred.shape == real.shape
        return self.coef1 * self.ce(logit, tgt) + self.coef2 * self.mse(pred, real)
