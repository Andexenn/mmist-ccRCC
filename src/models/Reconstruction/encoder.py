import os 
import logging 

import torch.nn as nn
import numpy as np 

logger = logging.getLogger(__name__)

class ReconstructEncoder(nn.Module):
    """ The encoder for reconstruct component in the pipeline """
    def __init__(self, wsi_feat, ct_feat, mri_feat, drop_out: float = 0.5):
        """
        Args:
            wsi_feat (x, 768): WSI feature
            ct_feat (x, 768): CT feature
            mri_feat (x, 768): MRI feature
            drop_out (float): regularization
        """

        self.wsi_feat = wsi_feat
        self.ct_feat = ct_feat
        self.mri_feat = mri_feat
        self.drop_out = drop_out 

    def encode(self):
        """
        Returns: feature with lower dimension
        """

        