"""Pytorch dataset object for MIL"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Dataloader

logger = logging.getLogger(__name__)

class FeatureBagDataset(Dataset):
    """Dataset for MIL components"""
    def __init__(self, clinical_file: str, feature_dir: str, modality: str = 'CT', split: str = 'train'):
        """
        Args:
            clinical_file (str): Containing the clinical infor
            feature_dir (str): The feature folder's path
            modality (str): 'CT' or 'MRI' or 'WSI'
            split (str): 'train' or 'test'
        """

        self.clinical_file = clinical_file
        self.feature_dir = feature_dir
        self.split = split
        self.modality = modality
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
                    f = f.unsqueeze(0)

                feature_list.append(f)
        else:
            logger.error("[ERR]:: %s does not exist.", bag_folder)

        if len(feature_list) > 0:
            features = torch.cat(feature_list)
            mask = 1
        else:
            feat_dim = 1024 if self.modality == 'WSI' else 512
            features = torch.zeros((1, feat_dim)).float()
            mask = 0

        return patient_id, features, label, torch.tensor(mask).long()


def get_mil_dataloader(
    clinical_file: str,
    feature_dir: str,
    modality: str,
    split: str,
    num_workers: int,
    batch_size: int = 1,
    shuffle: bool = True
):
    """MIL loader"""
    dataset = FeatureBagDataset(clinical_file, feature_dir, modality, split)

    loader = Dataloader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    return loader