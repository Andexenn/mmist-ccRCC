import os
import logging 
import glob 

import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

class ReconstructDataset(Dataset):
    """ 
    Load bags of all modality for each patient and then feed to MIL and select
    the best one to feed to reconstruction module with oversample of reconstruction module
    """

    def __init__(self, clinical_dir: str, feature_dir: str, split: str = 'train'):
        super().__init__()

        self.clinical_dir = clinical_dir
        self.feature_dir = feature_dir
        self.split = split

        full_df = pd.read_csv(self.clinical_dir)

        # Support combined train+val split for --stage test
        if split == 'train_val':
            split_df = full_df[full_df['Split'].isin(['train', 'val'])].reset_index(drop=True)
            # Track which original split each case came from (for file path resolution)
            self._original_splits = {}
            for _, row in full_df[full_df['Split'].isin(['train', 'val'])].iterrows():
                self._original_splits[row['case_id']] = row['Split']
        else:
            split_df = full_df[full_df['Split'] == split].reset_index(drop=True)
            self._original_splits = None

        # Build per-patient file_name lookup: {case_id: {modality: file_name}}
        self.file_name_lookup = {}
        if 'file_name' in split_df.columns:
            for _, row in split_df.iterrows():
                cid = row['case_id']
                mod = row.get('Modality', '')
                fname = row.get('file_name', '')
                if cid not in self.file_name_lookup:
                    self.file_name_lookup[cid] = {}
                self.file_name_lookup[cid][mod] = fname

        # Deduplicate to one row per patient for iteration
        self.df = split_df.drop_duplicates(subset='case_id').reset_index(drop=True)

        if len(self.df) == 0:
            logger.warning(
                "No data found for split=%s — dataset will be empty", split
            )
            return

        if split in ('train', 'train_val'):
            self._apply_oversample()

    def _apply_oversample(self):
        oversample_factor = 6

        minority_class = self.df[self.df['vital_status_12'] == 0]
        majority_class = self.df[self.df['vital_status_12'] == 1]

        if len(minority_class) > 0:
            minority_class = pd.concat([minority_class] * oversample_factor, ignore_index=True)
            self.df = pd.concat([minority_class, majority_class], ignore_index=True)
            self.df = self.df.sample(frac=1, random_state=42).reset_index(drop=True)
            logger.info(f"Reconstruction Dataset Oversampled (6x). New size: {len(self.df)}")


    @staticmethod
    def _safe_load_tensor(file_path: str) -> torch.Tensor:
        """Safely load a .pt file and ensure it returns a float tensor."""
        try:
            f = torch.load(file_path, weights_only=True)
        except Exception:
            try:
                f = torch.load(file_path, weights_only=False)
            except Exception as e:
                logger.warning("Failed to load %s: %s", file_path, e)
                return None

        if isinstance(f, dict):
            for v in f.values():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    f = v
                    break
            else:
                logger.warning("Loaded dict from %s but no tensor found", file_path)
                return None

        if isinstance(f, np.ndarray):
            f = torch.from_numpy(f).float()

        if not isinstance(f, torch.Tensor):
            try:
                f = torch.tensor(f).float()
            except Exception:
                logger.warning("Cannot convert object from %s (type=%s) to tensor", file_path, type(f).__name__)
                return None

        f = f.float()
        if f.dim() == 1:
            f = f.unsqueeze(0)
        return f

    def _load_bag(self, patient_id, modality):
        """Helper load list features of a modality"""
        # Try the new path structure: feature_dir/Modality/split/file_name
        file_name = None
        if patient_id in self.file_name_lookup:
            file_name = self.file_name_lookup[patient_id].get(modality)

        # Determine which split subdirectory to look in
        if self.split == 'train_val':
            # For combined split, resolve the original split for this patient
            splits_to_try = []
            if self._original_splits and patient_id in self._original_splits:
                splits_to_try.append(self._original_splits[patient_id])
            # Fallback: try both
            for s in ['train', 'val']:
                if s not in splits_to_try:
                    splits_to_try.append(s)
        else:
            splits_to_try = [self.split]

        feature_list = []

        for split_dir in splits_to_try:
            if file_name:
                bag_path = os.path.join(self.feature_dir, modality, split_dir, file_name)
            else:
                # Fallback for legacy structure
                bag_path = os.path.join(self.feature_dir, patient_id)

            try:
                if os.path.isdir(bag_path):
                    file_paths = sorted(glob.glob(os.path.join(bag_path, "*.pt")))
                    for fp in file_paths:
                        f = self._safe_load_tensor(fp)
                        if f is not None:
                            feature_list.append(f)
                elif os.path.isfile(bag_path) or os.path.isfile(bag_path + '.pt'):
                    fpath = bag_path if os.path.isfile(bag_path) else bag_path + '.pt'
                    f = self._safe_load_tensor(fpath)
                    if f is not None:
                        feature_list.append(f)
            except Exception as e:
                logger.error("Error loading bag for patient=%s modality=%s: %s", patient_id, modality, e)

            if feature_list:
                break  # Found data, no need to try other splits

        if len(feature_list) > 0:
            return feature_list, 1
        else:
            return [torch.zeros((1, 768)).float()], 0
        
    def _load_clinical(self, patient_id):
        return self._load_bag(patient_id, 'Clinical')
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        pid = row['case_id']

        wsi_bag, wsi_mask = self._load_bag(pid, 'WSI')
        ct_bag, ct_mask = self._load_bag(pid, 'CT')
        mri_bag, mri_mask = self._load_bag(pid, 'MRI')
        
        cli_bag, cli_mask = self._load_clinical(pid)
        cli_feat = cli_bag[0] 
        
        return {
            'wsi_bag': wsi_bag, 'wsi_mask': wsi_mask,
            'ct_bag': ct_bag,   'ct_mask': ct_mask,
            'mri_bag': mri_bag, 'mri_mask': mri_mask,
            'cli_feat': cli_feat, 'cli_mask': cli_mask,
            'label': torch.tensor(int(row['vital_status_12'])).long()
        }
    
def collate(batch):
    """Custom collate that handles variable-length bags for batch_size >= 1"""
    if len(batch) == 1:
        return batch[0]
    
    # For batch_size > 1, return list of sample dicts
    return batch

def get_reconstruct_dataloader(
    clinical_file: str, 
    feature_dir: str,
    split: str,
    num_workers: int = 4,
    batch_size: int = 1,
    shuffle: bool = True 
):
    """ Reconstruction loader """
    dataset = ReconstructDataset(clinical_file, feature_dir, split)

    loader = DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True 
    )

    return loader
