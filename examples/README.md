# Fetal Head Segmentation - Examples

This directory contains example scripts demonstrating various features of the project.

## Available Examples

### 1. `train_with_caching.py`
**Complete training pipeline with caching and logging**

Features demonstrated:
- CachedHC18Dataset for faster training
- TrainingLogger for comprehensive metric tracking
- PredictionSaver for saving model outputs
- Full training loop with checkpointing

Usage:
```bash
python examples/train_with_caching.py
```

---

## Quick Reference

### Basic Training (No Caching)
```python
from src.data import HC18Dataset
from torch.utils.data import DataLoader

dataset = HC18Dataset(
    image_dir='dataset/training_set/images',
    mask_dir='dataset/training_set/masks',
    transform=transforms
)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
```

### Training with Cache
```python
from src.data import CachedHC18Dataset

dataset = CachedHC18Dataset(
    image_dir='dataset/training_set/images',
    mask_dir='dataset/training_set/masks',
    cache_dir='preprocessed_data/train_cache',
    img_height=256,
    img_width=256,
    transform=transforms
)
```

### Logging Metrics
```python
from src.utils import TrainingLogger

logger = TrainingLogger(experiment_name='my_experiment')
logger.log_epoch(epoch, train_loss, train_dice, val_loss, val_dice, val_iou, val_pa, lr)
logger.save_checkpoint(epoch, model, optimizer, scheduler, val_dice, is_best=True)
logger.plot_metrics()
```

### Saving Predictions
```python
from src.utils import PredictionSaver

saver = PredictionSaver(experiment_name='my_experiment')
saver.save_from_dataloader(model, test_loader, device)
```

---

For detailed documentation, see `docs/CACHING_AND_LOGGING.md`.
