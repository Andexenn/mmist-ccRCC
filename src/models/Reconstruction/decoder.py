import os
import logging

import torch.nn as nn
import torch
import numpy as np

from models.Reconstruction.MLP_block import MLP

logger = logging.getLogger(__name__)

class ReconstructDecoder(nn.Module):
    """ The decoder part of reconstruct module """

    def __init__(self, input_dim: int = 128, hidden_dim: int = 128):
        super().__init__()

        logger.info('Reconstruct decoder is starting...')

        self.mlp_module = MLP(input_dim, [hidden_dim, hidden_dim])

    def forward(self, feature: torch.Tensor):
        if self.training:
            noise = torch.randn_like(feature) * 0.01
            feature = feature + noise

        decoded_feature = self.mlp_module(feature)
        return decoded_feature
        