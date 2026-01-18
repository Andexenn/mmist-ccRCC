"""Pytorch dataset object for MIL"""

import os
from typing import List
import glob
import logging

import numpy as np
import pandas as pd  
import torch 
from rich import print 
from torch.utils.data import Dataset, Dataloader

logger = logging.getLogger(__name__)

logger = setup_logger()

class FeatureBagDataset(Dataset):
    """Dataset for MIL components"""
    def __init__(self, clinical_file: str, feature_dir, modality: str = 'CT', split: str = 'train'):
        """
        Args:
            csv_file (str): Path to CSV containing 'patient_id', 'label', 'split'.
            data_dirs (dict): Dictionary mapping modality names to folder paths.
                              e.g. {'wsi': './features/wsi', 'ct': './features/ct'}
            Split (str): 'train', 'val', or 'test'.
        """

        self.clinical_file = clinical_file
        self.feature_dir = feature_dir
        self.split = split
        df = pd.read_csv(self.clinical_file)
        self.df = df[df['Split'] == split].reset_index(drop=True)
        self.df = self.df[self.df['Modality'] == modality].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['case_id']
        label = torch.tensor(int(row['vital_status_12'])).long()
        
        bag_folder = os.path.join(self.feature_dir, patient_id)

        feature_list = []

        if os.path.exists(bag_folder):
            file_paths = sorted(glob.glob(os.path.join(bag_folder, "*.pt")))

            for file_path in file_paths:
                f = torch.load(file_path)

                # change to tensort
                if isinstance(f, np.ndarray):
                    f = torch.from_numpy(f).float()

                if f.dim() == 1:
                    f = f.unsquueze(0)

                feature_list.append(f)
        else:
            logger.info(f"{bag_folder} does not exist.")

        if len(feature_list) > 0:
            features = torch.cat(feature_list)
            mask = 1 
        else:
            feat_dim = 1024 if 'wsi' else 512
            features = torch.zeros((1, feat_dim)).float()
            mask = 0

        return features, label, torch.tensor(mask).long()