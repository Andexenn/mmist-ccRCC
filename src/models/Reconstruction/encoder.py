import os
import logging

import torch.nn as nn
import torch
import numpy as np

from models.Reconstruction.MLP_block import MLP

logger = logging.getLogger(__name__)

class ReconstructEncoder(nn.Module):
    """ The encoder for reconstruct component in the pipeline """
    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        """
        Args:
            modality_feature
            drop_out (float): regularization
        """
        super().__init__()
        self.mlp_module = MLP(input_dim, [hidden_dim, hidden_dim])            

    def forward(self, feature: torch.Tensor):
        """
        Returns: feature with lower dimension
        """ 

        if self.training:
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise

        encoded_feature = self.mlp_module(feature)
        return encoded_feature

