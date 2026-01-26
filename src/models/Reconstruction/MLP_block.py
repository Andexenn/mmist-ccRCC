import os
import logging
from typing import List

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class MLP(nn.Module):
    """ Implement MLP block for reconstruction """

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        prev_dim = input_dim

        layers = []

        for hidden_dim in hidden_dims[:-1]:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, hidden_dims[-1]))

        self.mlp = nn.Sequential(*layers)

    def forward(self, feature: torch.Tensor):
        """ Feed the feature to MLP block and run it """

        compressed_feature = self.mlp(feature)
        return compressed_feature
