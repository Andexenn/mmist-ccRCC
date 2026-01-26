import os
import logging
from typing import List

import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import random

from models.Reconstruction.encoder import ReconstructEncoder
from models.Reconstruction.decoder import ReconstructDecoder
from models.Reconstruction.cross_modal import CrossModalLayer

logger = logging.getLogger(__name__)

class ReconstructionModel(nn.Module):
    """ Reconstruction module """
    def __init__(self, feature_dim: int = 768, hidden_dim: int = 128):
        super().__init__()
        
        self.wsi_encoder = ReconstructEncoder(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.mri_encoder = ReconstructEncoder(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.ct_encoder = ReconstructEncoder(input_dim=feature_dim, hidden_dim=hidden_dim)
        self.clinic_encoder = ReconstructEncoder(input_dim=feature_dim, hidden_dim=hidden_dim) 

        self.cross_modal = CrossModalLayer(num_modalities=4, encoder_dim=hidden_dim, output_dim=hidden_dim)

        self.wsi_decoder = ReconstructDecoder(input_dim=hidden_dim, output_dim=feature_dim)
        self.mri_decoder = ReconstructDecoder(input_dim=hidden_dim, output_dim=feature_dim)
        self.ct_decoder = ReconstructDecoder(input_dim=hidden_dim, output_dim=feature_dim)
        self.clinic_decoder = ReconstructDecoder(input_dim=hidden_dim, output_dim=feature_dim)


    def forward(self, wsi_feat: torch.Tensor, ct_feat: torch.Tensor, mri_feat: torch.Tensor, clinic_feat: torch.Tensor):

        wsi_emb = self.wsi_encoder(wsi_feat)
        ct_emb = self.ct_encoder(ct_feat)
        mri_emb = self.mri_encoder(mri_feat)
        cli_emb = self.clinic_encoder(clinic_feat)

        if self.training:
            embeddings = [wsi_emb, ct_emb, mri_emb, cli_emb]
            available_indices = [
                i for i, emb in enumerate(embeddings) 
                if torch.sum(torch.abs(emb)).item() > 1e-6
            ]

            if len(available_indices) > 1:
                drop_idx = random.choice(available_indices)
                embeddings[drop_idx] = torch.zeros_like(embeddings[drop_idx])
                
                wsi_emb, ct_emb, mri_emb, cli_emb = embeddings

        latent_vector = self.cross_modal([wsi_emb, mri_emb, ct_emb, cli_emb])

        #TODO: watch again, how to separate those features to 4 modalities
        rec_wsi = self.wsi_decoder(latent_vector)
        rec_ct = self.ct_decoder(latent_vector)
        rec_mri = self.mri_decoder(latent_vector)
        rec_cli = self.clinic_decoder(latent_vector)

        return rec_wsi, rec_ct, rec_mri, rec_cli
