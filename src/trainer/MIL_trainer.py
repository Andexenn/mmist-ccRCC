import os 
import logging
import copy

import numpy as np 
import sklearn
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)

def train_single_modality(
    mil_model,
    modality: str,
    train_loader,
    val_loader,
    output_dir,
    n_epochs, 
    device,
    lr: float,
    death_weight: float = 1.0
): 
    """ Train a single modality (WSI, CT, MRI) """
    logger.info("=== Training Modality: %s ===", modality)
    
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs', modality))

    if modality == 'WSI':
        sub_model = mil_model.wsi_mil
    elif modality == 'CT':
        sub_model = mil_model.ct_mil
    elif modality == 'MRI':
        sub_model = mil_model.mri_mil
    else:
        raise ValueError(f"{modality} is not valid")

    #AdamW
    optimizer = optim.AdamW(sub_model.parameters(), lr=lr, weight_decay=1e-5)

    # LR Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    criterion = nn.BCELoss(reduction='none')

    #Early Stopping
    patience_limit = 20
    patience_counter = 0
    best_val_loss = float('inf')
    best_model_state = None 

    for epoch in range(n_epochs):
        #training phase
        mil_model.train()

        running_loss = 0.0
        train_batches = 0

        for patient_id, feature_list, label, mask, in train_loader:
            if mask.item() == 0:
                continue 

            feature_list = [f.to(device) for f in feature_list]
            label = label.float().to(device).unsqueeze(0)

            optimizer.zero_grad(set_to_none=True)

            prob, _, _ = mil_model.forward_single_bag(
                feature_list,
                modality=modality,
                add_noise=True 
            )

            loss_unreduced = criterion(prob, label)
            weight = death_weight if label.item() == 0 else 1.0
            loss = (loss_unreduced * weight).mean()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = running_loss / train_batches if train_batches > 0 else 0.0

        #validation phase
        mil_model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for patient_id, feature_list, label, mask in val_loader:
                if mask.item() == 0:
                    continue 

                feature_list = [f.to(device) for f in feature_list]
                label = label.float().to(device).unsqueeze(0)

                prob, _, _ = mil_model.forward_single_bag(
                    feature_list,
                    modality=modality,
                    add_noise=False 
                )

                loss = criterion(prob, label)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0

        logger.info(f"Epoch {epoch+1}/{n_epochs} - {modality} Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        writer.add_scalar('Loss/Train', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val', avg_val_loss, epoch)

        scheduler.step(avg_val_loss)
                
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss 
            patience_counter = 0
            best_model_state = copy.deepcopy(sub_model.state_dict())

            save_path = os.path.join(output_dir, f'best_mil_{modality}.pth')
            torch.save(mil_model.state_dict(), save_path)
            logger.info(f"  -> Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience_limit:
                logger.info(f"Early stopping triggered for {modality} at epoch {epoch+1}")
                break

    writer.close()

    if best_model_state is not None:
        sub_model.load_state_dict(best_model_state)
        logger.info(f"Restored best weights for {modality}")


def train_mil_survival(mil_model, output_dir, n_epochs: int = 100, lr: float = 1e-3):
    """ 
    Orchestrator to train WSI, CT, and MRI MIL models sequentially.
    Strategy: Train separately -> freeze others -> update weights
    """
    logger.info("Begin training MIL module (Stage 1)")
    os.makedirs(output_dir, exist_ok=True)        

    train_loaders_tuple = mil_model._get_dataloader(split='train', shuffle=True)
    val_loaders_tuple = mil_model._get_dataloader(split='test', shuffle=False)

    loaders = {
        'WSI': {'train': train_loaders_tuple[0], 'val': val_loaders_tuple[0]},
        'CT':  {'train': train_loaders_tuple[1], 'val': val_loaders_tuple[1]},
        'MRI': {'train': train_loaders_tuple[2], 'val': val_loaders_tuple[2]}
    }

    device = next(mil_model.parameters()).device 

    for modality in ['WSI', 'CT', 'MRI']:
        train_single_modality(
            mil_model=mil_model,
            modality=modality,
            train_loader=loaders[modality]['train'],
            val_loader=loaders[modality]['val'],
            output_dir=output_dir,
            n_epochs=n_epochs,
            device=device,
            lr=lr
        )
        
    logger.info("MIL Training Stage 1 Complete.")
