# MMIST Training Guide

This guide will instruct you on how to train your model in **Stage 1 (Train từng module)** and **Stage 2 (Finetune cả pipeline)** based on the workflow from your images. 

Your trainer code (`MIL_trainer.py`, `reconstruction_trainer.py`, `fusion_trainer.py`) **already has the correct logic to save checkpoints (lưu trọng số)** when the validation loss (`val loss`) is at its lowest. This guide shows you how to connect them in your `main.py` so you can easily run it on your server.

## 1. Stage 1: Train Separate Modules (Train từng module)

In this stage, you train the modules one by one. According to your image specs:
1. Train **MIL Selection** (100 epochs, AdamW lr=1e-3). 
2. Train **Missing Modality Reconstruction** (100 epochs, AdamW lr=1e-3). Freeze MIL.
3. Train **Multi-modal Fusion** (100 epochs, AdamW lr=1e-3). Freeze MIL and Reconstruction.

To implement this, you just need to wire them up. Here is what your `main.py` will look like:

```python
import torch

from models.MIL.model import MILModel
from models.Reconstruction.model import ReconstructionModel
from models.Fusion.model import Fusion

from trainer.MIL_trainer import train_mil_survival
from trainer.reconstruction_trainer import train_reconstruction_module
from trainer.fusion_trainer import train_fuse_module
from trainer.pipeline_trainer import train_pipeline

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = './checkpoints'

    # 1. Initialize models (Khởi tạo các mô hình)
    # NOTE: You need to pass your specific configs (cấu hình) here when initializing 
    mil_model = MILModel(...).to(device)
    recon_model = ReconstructionModel(...).to(device)
    fusion_model = Fusion(...).to(device)

    # ==========================================
    # STAGE 1: TRAIN TỪNG MODULE
    # ==========================================
    
    # Step 1: Train MIL
    # Checkpoints will automatically be saved as best_mil_WSI.pth, best_mil_CT.pth, best_mil_MRI.pth
    train_mil_survival(mil_model, output_dir=output_dir, n_epochs=100, lr=1e-3)

    # Step 2: Train Reconstruction
    # Checkpoint automatically saved as best_reconstruction.pth
    # Wait, before training the next module, load the best MIL weights first!
    mil_model.load_state_dict(torch.load(f"{output_dir}/best_mil_MRI.pth"))

    train_reconstruction_module(
        mil_model=mil_model, 
        recon_model=recon_model, 
        clinical_file='path/to/clinical.csv',
        feature_dir='path/to/features',
        output_dir=output_dir,
        n_epochs=100, 
        lr=1e-3
    )
    
    # Step 3: Train Fusion
    # Checkpoint automatically saved as best_fusion_model.pth
    recon_model.load_state_dict(torch.load(f"{output_dir}/best_reconstruction.pth"))
    
    train_fuse_module(
        mil_model=mil_model, 
        recon_model=recon_model, 
        fusion_model=fusion_model,
        clinical_file='path/to/clinical.csv',
        feature_dir='path/to/features',
        output_dir=output_dir,
        epochs=100, 
        lr=1e-3
    )
```

## 2. Stage 2: Finetune Pipeline (Finetune cả pipeline)

After successfully executing Stage 1, you will have `best_fusion_model.pth`, `best_reconstruction.pth`, and `best_mil_MRI.pth` saved securely in your `output_dir`.

In Stage 2, you need to load these saved checkpoints ("Sau khi có tất cả checkpoint cho từng module thì em sẽ tiến hành finetune...") and update (cập nhật) all of them together with `lr=1e-5`.

You can append this directly to the bottom of the `main()` function in your `main.py`:

```python
    # ==========================================
    # STAGE 2: FINETUNE ENTIRE PIPELINE
    # ==========================================
    
    # 1. Load the best checkpoints from Stage 1 (Tải trọng số tốt nhất từ Giai đoạn 1)
    mil_model.load_state_dict(torch.load(f"{output_dir}/best_mil_MRI.pth"))
    recon_model.load_state_dict(torch.load(f"{output_dir}/best_reconstruction.pth"))
    fusion_model.load_state_dict(torch.load(f"{output_dir}/best_fusion_model.pth"))

    # 2. Start Stage 2 Finetuning (Bắt đầu finetune)
    # The pipeline_trainer will use AdamW with lr=1e-5 and unfreeze (mở khóa) all parameters automatically.
    # The final Checkpoint will be saved as best_pipeline.pth
    train_pipeline(
        mil_model=mil_model,
        recon_model=recon_model,
        fusion_model=fusion_model,
        clinical_file='path/to/clinical.csv',
        feature_dir='path/to/features',
        output_dir=output_dir,
        epochs=100,
        lr=1e-5
    )

if __name__ == "__main__":
    main()
```

### Checkpoints Summary (Tóm tắt về các file lưu trọng số)
Bạn không cần phải code thêm phần lưu checkpoints vì mình đã kiểm tra và thấy logic lưu trong code của bạn đã bao gồm đủ:
* **Stage 1 (MIL)**: Saves `best_mil_{modality}.pth` (e.g. `best_mil_MRI.pth`) based on the lowest validation loss of the sub-models.
* **Stage 1 (Recon)**: Saves `best_reconstruction.pth` based on the lowest validation Mean Squared Error (MSE).
* **Stage 1 (Fusion)**: Saves `best_fusion_model.pth` based on Binary Cross Entropy (BCE) validation loss.
* **Stage 2 (Pipeline)**: Saves `best_pipeline.pth` (which contains a Python Dictionary packing all 3 state dicts together) based on end-to-end BCE validation loss.
