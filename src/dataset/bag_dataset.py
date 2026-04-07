"""Pytorch dataset object for MIL"""

import os
import glob
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class FeatureBagDataset(Dataset):
    """Dataset for MIL components"""
    def __init__(self, clinical_file: str, feature_dir: str, modality: str = 'CT', split: str = 'train'):
        """
        Args:
            clinical_file (str): Containing the clinical infor
            feature_dir (str): The feature folder's path
            modality (str): 'CT' or 'MRI' or 'WSI'
            split (str): 'train', 'test', 'val', or 'train_val' (combined train+val)
            type (str): 'MIL' or 'Reconstruction' or 'Fusion'
        """

        self.clinical_file = clinical_file
        self.feature_dir = feature_dir
        self.split = split
        self.modality = modality
        df = pd.read_csv(self.clinical_file)

        # Support combined train+val split for --stage test
        if split == 'train_val':
            self.df = df[df['Split'].isin(['train', 'val'])].reset_index(drop=True)
            self.df = self.df[self.df['Modality'] == modality].reset_index(drop=True)
            # Track original splits for file path resolution
            self._original_splits = {}
            for _, row in df[df['Split'].isin(['train', 'val'])].iterrows():
                self._original_splits[row['case_id']] = row['Split']
        else:
            self.df = df[df['Split'] == split].reset_index(drop=True)
            self.df = self.df[self.df['Modality'] == modality].reset_index(drop=True)
            self._original_splits = None

        if len(self.df) == 0:
            logger.warning(
                "No data found for modality=%s, split=%s — dataset will be empty",
                modality, split
            )
            return

        if split in ('train', 'train_val'):
            self._apply_oversample()


    def _apply_oversample(self):
        """ 
        Oversampling minority class (death at 12 months)
        Per-modality factors (Table A.1): CT=8x, MRI=16x, WSI=8x
        """

        oversample_factors = {'CT': 8, 'MRI': 16, 'WSI': 8}
        oversample_factor = oversample_factors.get(self.modality, 8)

        minority_class = self.df[self.df['vital_status_12'] == 0]  # died = minority
        majority_class = self.df[self.df['vital_status_12'] == 1]  # survived = majority

        minority_oversampled = pd.concat([minority_class] * oversample_factor, ignore_index = True)

        self.df = pd.concat([majority_class, minority_oversampled], ignore_index=True)
        self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(
            "Oversampling for %s with %dx at minority class",
            self.modality, oversample_factor
        )

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _safe_load_tensor(file_path: str) -> torch.Tensor:
        """Safely load a .pt file and ensure it returns a float tensor.

        Handles: plain tensors, numpy arrays, dicts (extracts first tensor value),
        and any other unexpected types.
        """
        try:
            f = torch.load(file_path, weights_only=True)
        except Exception:
            # weights_only=True may reject some files; retry without it
            try:
                f = torch.load(file_path, weights_only=False)
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path, e)
                return None

        # Handle dict (e.g. {'features': tensor, ...})
        if isinstance(f, dict):
            for v in f.values():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    f = v
                    break
            else:
                logger.warning("Loaded dict from %s but no tensor found, keys=%s", file_path, list(f.keys()))
                return None

        # Handle numpy array
        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f).float()

        # Handle non-tensor types
        if not isinstance(f, torch.Tensor):
            try:
                f = torch.tensor(f).float()
            except Exception:
                logger.warning("Cannot convert loaded object from %s (type=%s) to tensor", file_path, type(f).__name__)
                return None

        f = f.float()
        if f.dim() == 1:
            f = f.unsqueeze(0)
        return f

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_id = row['case_id']
        label = torch.tensor(int(row['vital_status_12'])).long()

        # Determine which split directories to try for file loading
        if self.split == 'train_val':
            splits_to_try = []
            if self._original_splits and patient_id in self._original_splits:
                splits_to_try.append(self._original_splits[patient_id])
            for s in ['train', 'val']:
                if s not in splits_to_try:
                    splits_to_try.append(s)
        else:
            splits_to_try = [self.split]

        try:
            feature_list = []
            for split_dir in splits_to_try:
                # Use file_name + modality + split to build the correct path
                # Structure: feature_dir/Modality/Split/file_name
                if 'file_name' in row.index:
                    bag_folder = os.path.join(
                        self.feature_dir, self.modality, split_dir, row['file_name']
                    )
                else:
                    # Fallback for legacy CSVs without file_name column
                    bag_folder = os.path.join(self.feature_dir, patient_id)

                if os.path.isdir(bag_folder):
                    # Entry is a folder containing .pt files
                    file_paths = sorted(glob.glob(os.path.join(bag_folder, "*.pt")))
                    for file_path in file_paths:
                        f = self._safe_load_tensor(file_path)
                        if f is not None:
                            feature_list.append(f)
                elif os.path.isfile(bag_folder) or os.path.isfile(bag_folder + '.pt'):
                    # Entry is a single .pt file
                    fpath = bag_folder if os.path.isfile(bag_folder) else bag_folder + '.pt'
                    f = self._safe_load_tensor(fpath)
                    if f is not None:
                        feature_list.append(f)

                if feature_list:
                    break  # Found data, no need to try other splits

            if not feature_list:
                logger.error("[ERR]:: No data found for patient=%s in any split dir.", patient_id)

        except Exception as e:
            logger.error("Unexpected error loading sample idx=%d (patient=%s): %s", idx, patient_id, e)
            feature_list = []

        if len(feature_list) > 0:
            mask = 1
        else:
            feature_list = [torch.zeros((1, 768)).float()]
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

    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_mil,
        pin_memory=True
    )

    return loader