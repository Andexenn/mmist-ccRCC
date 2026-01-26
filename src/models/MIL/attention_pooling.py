import logging 
import os 

import torch.nn as nn
import torch.F as F
import torch
import numpy as np 
import pandas as pd 

logger = logging.getLogger(__name__)

class AttentionPooling(nn.Module):
    """
    Aggregates patches (N, D) into a single image vector (1, D) using learnable weights V and w
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.V = nn.Linear(input_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        v_out = torch.tanh(self.V(h))
        scores = self.w(v_out)
        weights = F.softmax(scores, dim=0)
        aggregate_feature = torch.sum(weights * h, dim=0, keepdim=True)

        return aggregate_feature

