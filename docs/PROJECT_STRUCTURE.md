# Project Structure (Updated)

## Overview

This project is organized into two main development stages:

1. **Accuracy Focus** - Maximizing segmentation accuracy (current stage)
2. **Efficient Focus** - Optimizing for real-time performance (future stage)

## Directory Structure

```
Fetal Head Segmentation/
│
├── shared/                          # Shared resources across all implementations
│   ├── src/
│   │   ├── data/                   # Dataset loaders and preprocessing
│   │   ├── losses/                 # Loss functions (Dice, BCE, etc.)
│   │   ├── metrics/                # Evaluation metrics (DSC, mIoU, mPA)
│   │   └── utils/                  # Helper utilities
│   ├── configs/                    # Shared configuration files
│   ├── dataset/                    # Original HC18 dataset
│   └── prepare_dataset.py          # Dataset preparation script
│
├── accuracy_focus/                 # CURRENT STAGE: High accuracy models
│   │
│   ├── standard_u_net/             # Baseline Standard U-Net
│   │   ├── src/
│   │   │   └── models/            # Standard U-Net architecture
│   │   ├── configs/               # Model-specific configs
│   │   ├── results/               # Training outputs
│   │   │   ├── checkpoints/       # Model checkpoints
│   │   │   ├── logs/              # Training logs
│   │   │   ├── predictions/       # Test predictions
│   │   │   └── visualizations/    # Result visualizations
│   │   ├── weights/               # Trained model weights
│   │   ├── preprocessed_data/     # Cached preprocessed data
│   │   │   ├── train_cache/
│   │   │   ├── val_cache/
│   │   │   └── test_cache/
│   │   └── train.py               # Training script
│   │
│   └── improved_u_net/             # Enhanced U-Net (ResBlocks, ASPP, FP+SAM)
│       ├── src/
│       │   └── models/            # Improved U-Net architecture
│       ├── configs/               # Model-specific configs
│       ├── results/               # Training outputs
│       ├── weights/               # Trained model weights
│       ├── preprocessed_data/     # Cached preprocessed data
│       └── train.py               # Training script (to be created)
│
├── efficient_focus/                # FUTURE STAGE: Optimized models
│   ├── standard_u_net/             # Optimized Standard U-Net
│   └── improved_u_net/             # Optimized Improved U-Net
│
├── notebooks/                      # Jupyter notebooks for experiments
│   ├── accuracy_focus/
│   │   ├── standard_u_net/
│   │   └── improved_u_net/
│   └── efficient_focus/
│
├── docs/                           # Documentation
├── examples/                       # Example scripts
├── main.py                         # Main entry point
└── requirements.txt                # Python dependencies
```

## Key Principles

### Shared Resources

- **Dataset**: Single source of truth in `shared/dataset/`
- **Core modules**: Data loaders, losses, metrics, utilities in `shared/src/`
- **Common configs**: Base configurations in `shared/configs/`

### Model Separation

- Each model variant has its own:
  - Source code (`src/models/`)
  - Configuration files (`configs/`)
  - Training results (`results/`)
  - Trained weights (`weights/`)
  - Preprocessed data cache (`preprocessed_data/`)

### Stage Separation

- **accuracy_focus/**: Prioritizes segmentation accuracy (DSC ≥97.81%)
- **efficient_focus/**: Prioritizes inference speed (near real-time)

## Usage

### Training Standard U-Net

```bash
cd accuracy_focus/standard_u_net
python train.py
```

### Training Improved U-Net (when ready)

```bash
cd accuracy_focus/improved_u_net
python train.py
```

### Shared Utilities

All models import from `shared/src/`:

```python
from shared.src.data import FetalHeadDataset
from shared.src.losses import DiceLoss
from shared.src.metrics import calculate_metrics
```

## Migration Notes

- Old `src/` → Split into `shared/src/` and model-specific `src/`
- Old `configs/` → Split into `shared/configs/` and model-specific `configs/`
- Old `results/` → Moved to `accuracy_focus/standard_u_net/results/`
- Old `weights/` → Moved to `accuracy_focus/standard_u_net/weights/`
- Old `preprocessed_data/` → Moved to `accuracy_focus/standard_u_net/preprocessed_data/`
- Old `dataset/` → Moved to `shared/dataset/`
