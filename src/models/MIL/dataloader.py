"""Pytorch dataset object for MIL"""

import os
import numpy as np
import pandas as pd  
import torch 
from rich import print 
from torch.utils.data import Dataset, Dataloader

class MultiModalMILDataset(Dataset):
    def __init__(self, csv_file, data_dirs, split='train'):
        """
        Args:
            csv_file (str): Path to CSV containing 'patient_id', 'label', 'split'.
            data_dirs (dict): Dictionary mapping modality names to folder paths.
                              e.g. {'wsi': './features/wsi', 'ct': './features/ct'}
            split (str): 'train', 'val', or 'test'.
        """

        self.data_dirs = data_dirs
        
        # 1. Load and filter CSV
        df = pd.read_csv(csv_file)
        self.df = df[df['split'] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['patient_id']
        label = torch.tensor(int(row['label'])).long()
        
        data = {}
        masks = {}

        for modality, dir_path in self.data_dirs.items():
            feature_path = os.path.join(dir_path, f"{patient_id}.pt")

            if os.path.exists(feature_path):
                features = torch.load(feature_path)

                if isinstance(features, np.ndarray):
                    features = torch.from_numpy(features).float()

                mask = 1
            else:
                feat_dim = 1024 if modality == 'wsi' else 512 
                features = torch.zeros((1, feat_dim)).float()
                mask = 0

            data[modality] = features 
            masks[modality] = torch.tensor(mask).long()

        return data, label, masks 

