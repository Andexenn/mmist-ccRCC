import os 
from typing import List

import torch 
import torch.nn as nn
import numpy as np 
import pandas as pd 

from models.Fusion.binary_MLP import BinaryMLP
from utils.metrics import calc_bacc

class LateFusion(nn.Module):
    def __init__(self, input_dims: List, num_modalities: int): 
        super().__init__()
        self.classifiers = nn.ModuleList([
            BinaryMLP(input_dim)
            for input_dim in input_dims
        ])
        
        self.fusion_weights = nn.Parameter(torch.ones(num_modalities))

    def forward(self, feature_list, masks, mode: str = 'LW', BAcc_mods: List[torch.Tensor] = []):
        if mode == 'LW':
            return self._fuse_LW(feature_list, masks)
        elif mode == 'WS':
            return self._fuse_WS(feature_list, BAcc_mods, masks)

    def _fuse_WS(self, feature_list: List[torch.Tensor], BAcc_mods: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        numerator = 0
        denominator = 0

        for i, (feat, mask, BAcc) in enumerate(zip(feature_list, masks, BAcc_mods)):
            y_pred = self.classifiers[i](feat) 
            
            numerator += mask * BAcc * y_pred 
            denominator += mask * BAcc 

        p_survival = numerator / (denominator + 1e-6)

        return p_survival
    
    def _fuse_LW(self, feature_list: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        numerator = 0
        denominator = 0 

        learned_w = torch.sigmoid(self.fusion_weights)

        for i, (feat, mask) in enumerate(zip(feature_list, masks)):
            prob = self.classifiers[i](feat)

            w_m = learned_w[i]

            numerator += mask * w_m * prob 
            denominator += mask * w_m 
        
        p_survival = numerator / (denominator + 1e-6)

        return p_survival

