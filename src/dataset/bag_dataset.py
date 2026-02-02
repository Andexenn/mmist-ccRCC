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
            type (str): 'MIL' or 'Reconstruction' or 'Fusion'
        """

        self.clinical_file = clinical_file
        self.feature_dir = feature_dir
        self.split = split
        self.modality = modality
        df = pd.read_csv(self.clinical_file)
        self.df = df[df['Split'] == split].reset_index(drop=True)
        self.df = self.df[self.df['Modality'] == modality].reset_index(drop=True)

        if split == 'train':
            self._apply_oversample()


    def _apply_oversample(self):
        """ 
        Oversampling minority class (death at 12 months) with 6x 
        """

        oversample_factor = 6

        minority_class = self.df[self.df['vital_status_12'] == 1]
        majority_class = self.df[self.df['vital_status_12'] == 0]

        minority_oversampled = pd.concat([minority_class] * oversample_factor, ignore_index = True)

        self.df = pd.concat([majority_class, minority_oversampled], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            "Oversampling for %s with 6x at minority class",
            self.type
        )

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
            mask = 1
        else:
            feature_list = torch.zeros((1, 768)).float()
            mask = 0

        return patient_id, feature_list, label, torch.tensor(mask).long()

def collate_mil(batch):
    """
    Custom collate function to handle List of Tensors with variable shapes
    Since batch_size=1, just extract the first element.
    """
    elem = batch[0]
    patient_id, feature_list, label, mask = elem
    return patient_id, feature_list, label, mask

def get_mil_dataloader(
    clinical_file: str,
    feature_dir: str,
    modality: str,
    split: str,
    num_workers: int,
    batch_size: int = 1,
    shuffle: bool = True
):
    """ MIL loader """
    dataset = FeatureBagDataset(clinical_file, feature_dir, modality, split)

    loader = Dataloader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_mil,
        pin_memory=True
    )

    return loader