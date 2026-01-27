from typing import List, Any, Dict
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

from models.MIL.gated_attention_pooling import GatedAttentionPooling
from dataset.bag_dataset import get_mil_dataloader

logger = logging.getLogger(__name__)

class SingleMIL(nn.Module):
    """ Single MIL for each modality """
    def __init__(self, input_dim: int, hidden_dims: List):
        """
        Args:
            input_dim (int): The dimension of input
            hidden_dims (List): The dimensions of linear layers
            dropout (float): Regularization
        """
        super().__init__()

        self.patch_attention = GatedAttentionPooling(input_dim)

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, patient_images: List[torch.Tensor]):
        """
        Args:
            bag_faetures (List[torch.Tensors]): all instances of 1 modality for one patient

        Return:
            max_prob (float): highest survival probability among all instances
            max_features (feature_dim, ): the feature of the selected instance
        """

        image_embeds = []

        for img in patient_images:
            img = img.to(next(self.parameters()).device)
            compressed_img, _ = self.patch_attention(img)
            image_embeds.append(compressed_img)

        bag_features = torch.cat(image_embeds, dim=0)
        instances_probs = self.mlp(bag_features)

        max_ids = torch.argmax(instances_probs)
        max_probs = instances_probs[max_ids]
        max_features = bag_features[max_ids]

        return max_ids, max_probs, max_features 


class MILModel(nn.Module):
    """ MIL Model for feature selection as described in the paper """
    def __init__(
        self,
        feature_dir: str,
        clinical_dir: str,
        device: str = 'cuda',
        dim: int = 768,
        loss_fn: str = "weighted BCE"
    ):
        """
        Args:
            feature_dir (str): the feature directory's path
            clinical_dir (str): the csv clinical file's path
            device (str): 'cuda' or 'cpu'
            dim (int): the dimension of the feature
            loss_fn (str): use weighted binary cross entropy like paper
        """

        super().__init__()

        self.feature_dir = feature_dir
        self.clinical_dir = clinical_dir
        self.df = pd.read_csv(self.clinical_dir)
        self.dim = dim
        self.loss_fn = loss_fn

        self.wsi_mil = SingleMIL(dim, [512, 256, 128])
        self.mri_mil = SingleMIL(dim, [256, 128])
        self.ct_mil = SingleMIL(dim, [256, 128])

        self.to(device)

    def _get_dataloader(self, split: str = 'train', shuffle: bool = True):
        """ Return WSI, CT and MRI dataloader """
        WSI_dataloader = get_mil_dataloader(
            feature_dir=self.feature_dir,
            clinical_file=self.clinical_dir,
            modality='WSI',
            split=split,
            shuffle=shuffle,
            num_workers=4)

        CT_dataloader = get_mil_dataloader(
            feature_dir=self.feature_dir,
            clinical_file=self.clinical_dir,
            modality='CT',
            split=split,
            shuffle=shuffle,
            num_workers=4)

        MRI_dataloader = get_mil_dataloader(
            feature_dir=self.feature_dir,
            clinical_file=self.clinical_dir,
            modality='MRI',
            split=split,
            shuffle=shuffle,
            num_workers=4)

        return WSI_dataloader, CT_dataloader, MRI_dataloader

    def forward_single_bag(self, feature_list: List[torch.Tensor], modality:str = 'WSI', add_noise: bool = False):
        """ Compute the selected feature """
        if add_noise:
            feature_list = [f + (torch.rand_like(f) * 0.01) for f in feature_list]

        if modality == 'WSI':
            survival_prob, selected_feature, selected_idx = self.wsi_mil(feature_list)
        elif modality == 'MRI':
            survival_prob, selected_feature, selected_idx = self.mri_mil(feature_list)
        elif modality == 'CT':
            survival_prob, selected_feature, selected_idx = self.ct_mil(feature_list)
        else:
            logger.error('%s is not suitable', modality)
            raise ValueError(f'Unknown modality: {modality}')

        return survival_prob, selected_feature, selected_idx

    def select_best_features_for_case(self, case_id: str, split: str = 'train'):
        """
        Select the best features for each modality of that patient
        Args:
            case_id (str): id of the case
            split (str): 'train' or 'test'

        Returns:
            dict with selected feature for each modality
        """

        self.eval()
        dataloaders = self._get_dataloader(split, shuffle=False)

        results: Dict[str, Any] = {'WSI': None, 'MRI': None, 'CT': None}

        modality_map = {
            'WSI': dataloaders[0],
            'CT':  dataloaders[1],
            'MRI': dataloaders[2]
        }

        def process_modality(modality: str, loader):
            for patient_id, features_list, label, mask in loader:
                if patient_id == case_id:
                    if mask.item() == 1:
                        with torch.no_grad():
                            use_noise = (split == 'train')
                            prob, selected_feat, idx = self.forward_single_bag(features_list, modality, add_noise=use_noise)

                            return {
                                'features': selected_feat.cpu(),
                                'probability': prob.item(),
                                'selected_idx': idx.item(),
                                'label': label.item()
                            }
                    return None
            return None 

        for mod_name, loader in modality_map.items():
            res = process_modality(mod_name, loader)
            if res is not None:
                results[mod_name] = res
            else:
                 results[mod_name] = {
                    'features': torch.zeros((1, self.dim)),
                    'probability': 0.0,
                    'selected_index': -1,
                    'label': -1
                }
        return results
