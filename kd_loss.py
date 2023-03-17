"""
Reference:
1. https://github.com/DefangChen/SimKD/blob/2b389c31ed7779aea31e7aaf0bb0f2d8b6ac2f01/distiller_zoo/KD.py
"""

import torch.nn as nn
import torch.nn.functional as F


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, y_s, y_t):
        """
        Input: student predict label (y_s), teacher predict label (y_t)
        Output: loss (DL divergence)
        """
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)
        loss = nn.KLDivLoss(reduction='batchmean')(p_s, p_t) * (self.T**2)
        return loss