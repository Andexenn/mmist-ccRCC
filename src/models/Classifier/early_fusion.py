import torch 
import torch.nn as nn
from typing import List
from models.Classifier.binary_MLP import BinaryMLP

class EarlyFusion(nn.Module):
    def __init__(self, fusion_type: str = 'mean', input_dim: int = 768, num_modalities: int = 4):
        """
        Args:
            fusion_type (str): 'mean' or 'cat'
            input_dim (int): Dimension of each modality feature (e.g., 768)
            num_modalities (int): Total number of modalities (for calculating cat dim)
        """
        super().__init__()
        self.fusion_type = fusion_type
        
        if self.fusion_type == 'cat':
            classifier_input_dim = input_dim * num_modalities 
        else:
            classifier_input_dim = input_dim

        self.binary_mlp_module = BinaryMLP(input_dim=classifier_input_dim)


    def _fuse_cat(self, feature_list: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor: 
        masked_features = [feat * mask for feat, mask in zip(feature_list, masks)]
        
        feat_cat = torch.cat(masked_features, dim=1)
        
        p_survival = self.binary_mlp_module(feat_cat)
        return p_survival

    def _fuse_mean(self, feature_list: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor: 
        stack_feats = torch.stack(feature_list, dim=1) 
        stack_masks = torch.stack(masks, dim=1)
        
        numerator = torch.sum(stack_feats * stack_masks, dim=1)
        
        denominator = torch.sum(stack_masks, dim=1)
        
        feat_mean = numerator / (denominator + 1e-6)
        
        p_survival = self.binary_mlp_module(feat_mean)
        return p_survival
    
    def forward(self, feature_list: List[torch.Tensor], masks: List[torch.Tensor]) -> torch.Tensor:
        """
        feature_list: List các tensor [Batch, 768]. Modality thiếu có thể là tensor 0 hoặc bất kỳ (sẽ bị mask).
        masks: List các mask [Batch, 1] (1=có, 0=thiếu).
        """
        if self.fusion_type == 'cat':
            return self._fuse_cat(feature_list, masks)
        else:
            return self._fuse_mean(feature_list, masks)
        