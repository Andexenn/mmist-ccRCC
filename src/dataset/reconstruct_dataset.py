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
        
        self.df = pd.read_csv(self.clinical_dir)

        self.df = self.df[self.df['Split'] == split].reset_index(drop=True)

        if split == 'train':
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


    def _load_bag(self, patient_id, pattern):
        """Helper load list features of a modality"""
        bag_folder = os.path.join(self.feature_dir, patient_id)

        feature_list = []
        if os.path.exists(bag_folder):
            file_paths = sorted(glob.glob(os.path.join(bag_folder, pattern)))
            for fp in file_paths:
                f = torch.load(fp)
                if isinstance(f, np.ndarray):
                    f = torch.from_numpy(f).float()
                if f.dim() == 1:
                    f = f.unsqueeze(0)
                feature_list.append(f)

        if len(feature_list) > 0:
            return feature_list, 1
        else:
            return [torch.zeros((1, 768)).float()], 0
        
    def _load_clinical(self, patient_id):
        return self._load_bag(patient_id, "*clinical*.pt")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx): 
        row = self.df.iloc[idx]
        pid = row['case_id']

        wsi_bag, wsi_mask = self._load_bag(pid, "*wsi*.pt")
        ct_bag, ct_mask = self._load_bag(pid, "*ct*.pt")
        mri_bag, mri_mask = self._load_bag(pid, "*mri*.pt")
        
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
    return batch[0]

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
