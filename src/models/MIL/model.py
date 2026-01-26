from typing import List, Any, Dict
import logging

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd

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

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.GELU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.mlp = nn.Sequential(*layers)

    def forward(self, bag_features):
        """
        Args:
            bag_faetures (tensor): all instances of 1 modality for one patient

        Return:
            max_prob (float): highest survival probability among all instances
            max_features (feature_dim, ): the feature of the selected instance
        """

        instances_probs = self.mlp(bag_features)

        max_idx = torch.argmax(instances_probs)
        max_prob = instances_probs[max_idx]
        max_features = bag_features[max_idx]

        return max_prob, max_features, max_idx

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

        # init 3 mil system
        self.wsi_mil = SingleMIL(dim, [512, 256, 128])
        self.mri_mil = SingleMIL(dim, [256, 128])
        self.ct_mil = SingleMIL(dim, [256, 128])

        # move to device
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

    def forward_single_bag(self, bag_features, modality:str = 'WSI', add_noise: bool = False):
        """ Compute the selected feature """
        if add_noise and self.training:
            noise = torch.randn_like(bag_features) * 0.01
            bag_features += noise

        if modality == 'WSI':
            survival_prob, selected_feature, selected_idx = self.wsi_mil(bag_features)
        elif modality == 'MRI':
            survival_prob, selected_feature, selected_idx = self.mri_mil(bag_features)
        elif modality == 'CT':
            survival_prob, selected_feature, selected_idx = self.ct_mil(bag_features)
        else:
            logger.error('%s is not suitable', modality)
            raise ValueError(f'Unknown modality: {modality}')

        return survival_prob, selected_feature, selected_idx

    def select_best_features_for_case(self, case_id: str, split: str='train'):
        """
        Select the best features for each modality of that patient
        Args:
            case_id (str): id of the case
            split (str): 'train' or 'test'

        Returns:
            dict with selected feature for each modality
        """

        self.eval()
        WSI_dataloader, CT_dataloader, MRI_dataloader = self._get_dataloader(split, shuffle=False)

        results: Dict[str, Any] = {
            'WSI': torch.zeros(),
            'MRI': torch.zeros(),
            'CT': torch.zeros()
        }

        # Process WSI
        for patient_id, features, label, mask in WSI_dataloader:
            if patient_id[0] == case_id:
                # features shape: (1, num_instances, feature_dim)
                features = features.squeeze(0).to(self.device)  # (num_instances, feature_dim)

                if mask.item() == 1:  # Valid features exist
                    with torch.no_grad():
                        prob, selected_feat, idx = self.forward_single_bag(features, 'WSI', add_noise=False)

                    results['WSI'] = {
                        'features': selected_feat.cpu(),
                        'probability': prob.item(),
                        'selected_index': idx.item(),
                        'label': label.item()
                    }
                break

        # Process CT
        for patient_id, features, label, mask in CT_dataloader:
            if patient_id[0] == case_id:
                features = features.squeeze(0).to(self.device)

                if mask.item() == 1:
                    with torch.no_grad():
                        prob, selected_feat, idx = self.forward_single_bag(features, 'CT', add_noise=False)

                    results['CT'] = {
                        'features': selected_feat.cpu(),
                        'probability': prob.item(),
                        'selected_index': idx.item(),
                        'label': label.item()
                    }
                break

        # Process MRI
        for patient_id, features, label, mask in MRI_dataloader:
            if patient_id[0] == case_id:
                features = features.squeeze(0).to(self.device)

                if mask.item() == 1:
                    with torch.no_grad():
                        prob, selected_feat, idx = self.forward_single_bag(features, 'MRI', add_noise=False)

                    results['MRI'] = {
                        'features': selected_feat.cpu(),
                        'probability': prob.item(),
                        'selected_index': idx.item(),
                        'label': label.item()
                    }
                break

        return results
