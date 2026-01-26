import os
import logging
from typing import List

import torch.nn as nn
import torch
import numpy as np


class CrossModalLayer(nn.Module):
    """ CrossModalLayer for reconstructing missing modality """

    def __init__(self, num_modalities: int = 4, encoder_dim: int = 128, output_dim: int = 128):
        """ 
        Args:
            num_modalities (int): the number of modalities are fed
            encoder_dim (int): the dimension of the encoder
            output_dim (int): the output dimension 
        """
        super().__init__()

        self.input_dim = num_modalities * encoder_dim
        self.layer = nn.Sequential(
            nn.Linear(self.input_dim, output_dim),
            nn.GELU()
        )

    def forward(self, feature_list: List[torch.Tensor]):
        """
        Args:
            feature_list (List[torch.Tensor]): Including all representative vector of each modality
        Returns:
            fused_feature: After concat and put through fc layer
        """

        concat_feature = torch.cat(feature_list, dim=1)
        fused_feature = self.layer(concat_feature)

        return fused_feature



