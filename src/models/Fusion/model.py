import torch 
import torch.nn as nn
from typing import List

from models.Fusion.early_fusion import EarlyFusion
from models.Fusion.late_fusion import LateFusion

class Fusion(nn.Module):
    def __init__(self, 
                 fusion_strategy: str,
                 input_dims: List[int],
                 num_modalities: int = 4):
        """
        Args:
            fusion_strategy (str): 'early_mean', 'early_cat', 'late_ws', 'late_lw'
            input_dims (List[int]): List of input dimensions for each modality. 
                                    (e.g., [768, 768, 768, 768])
            num_modalities (int): Total number of modalities.
        """
        super().__init__()
        self.fusion_strategy = fusion_strategy
        self.input_dims = input_dims
        self.num_modalities = num_modalities

        if self.fusion_strategy.startswith('early'):
            ef_type = self.fusion_strategy.split('_')[1] 
            
            self.fusion_module = EarlyFusion(
                fusion_type=ef_type, 
                input_dim=self.input_dims[0], 
                num_modalities=self.num_modalities
            )
            
        elif self.fusion_strategy.startswith('late'):
            self.fusion_module = LateFusion(
                input_dims=self.input_dims, 
                num_modalities=self.num_modalities
            )
            
        else:
            raise ValueError(f"Unknown fusion strategy: {self.fusion_strategy}")

    def forward(self, 
                feature_list: List[torch.Tensor], 
                masks: List[torch.Tensor], 
                bacc_mods: List[float]) -> torch.Tensor:
        """
        Unified forward pass that delegates to the specific module.
        """
        
        if self.fusion_strategy.startswith('early'):
            return self.fusion_module(feature_list, masks)

        elif self.fusion_strategy.startswith('late'):
            lf_mode = 'LW' if 'lw' in self.fusion_strategy else 'WS'
            
            return self.fusion_module(
                feature_list=feature_list, 
                masks=masks, 
                mode=lf_mode, 
                BAcc_mods=bacc_mods
            )
            
        return None
    