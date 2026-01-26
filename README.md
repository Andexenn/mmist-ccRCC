# Code for running pipeline in paper "MMIST-ccRCC: A Real World Medical Dataset for the Development of Multi-Modal Systems"

I do this because the paper didn't provide source code.

1. Folder structure

```
src/
├── configs
├── dataset
├── models/
│   ├── __init__.py
│   ├── mil/
│   │   ├── __init__.py
│   │   ├── dataloader.py
│   │   ├── main.py
│   │   ├── model.py
│   │   └── mnist_bag_loader.py
│   ├── modality_reconstruction/
│   │   └── __init__.py
│   └── multimodal_fusion/
│       ├── __init__.py
│       ├── early_fusion.py
│       └── late_fusion.py
├── main.py
├── utils/
│   ├── __init__.py
│   └── metrics.py
└── trainer/
    ├── __init__.py
    └── survival_trainer.py
README.md
requirements.txt
```

```
src
  configs
  data
    __init__.py
    base_dataset.py
    mil_dataset.py
    fusion_dataset.py
    transform.py
  models
    __init__.py
    mil
      __init__.py
      main.py
      model.py
      mnist_bag_loader.py
    modality_reconstruction
      __init__.py
    multimodal_fusion
      __init__.py
      early_fusion.py
      late_fusion.py
  main.py
  utils
    __init__.py
    metrics.py
  trainer
    __init__.py
    survival_trainer.py
README.md
requirements.txt
```