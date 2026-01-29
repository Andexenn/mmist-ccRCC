import os 

import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd 

class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int = 128):
        """
        Args:
            input_dim (int): The input dimension
        """
        super().__init__()

        self.mlp = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, input_dim),
            nn.GELU(), 
            # layer 2
            nn.Linear(input_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, mod_feature: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mod_feature (torch.Tensor): The modality's feature after reconstruting
        Returns:
            prob (torch.Tensor): The 12 month survival probability, value range [0, 1]
        """
        prob = self.mlp(mod_feature)

        return prob
