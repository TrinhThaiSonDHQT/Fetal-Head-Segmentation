"""
Quick Reference: Caching and Logging Usage

Copy-paste code snippets for common tasks.

Note: Add parent directory to Python path if running from examples/:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
"""

# ============================================================================
# 1. BASIC TRAINING WITH CACHING
# ============================================================================

from src.data import CachedHC18Dataset
from src.utils import get_transforms
from torch.utils.data import DataLoader

# Create cached dataset
train_dataset = CachedHC18Dataset(
    image_dir='dataset/training_set/images',
    mask_dir='dataset/training_set/masks',
    cache_dir='preprocessed_data/train_cache',
    img_height=256,
    img_width=256,
    transform=get_transforms(256, 256, is_train=True),
    force_rebuild=False  # Set True to rebuild cache
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)


# ============================================================================
# 2. INITIALIZE LOGGER
# ============================================================================

from src.utils import TrainingLogger

logger = TrainingLogger(
    experiment_name='improved_unet_exp1',
    base_dir='results',
    save_plots=True
)

# Save configuration
config = {
    'model': 'ImprovedUNet',
    'img_size': 256,
    'batch_size': 8,
    'learning_rate': 0.1,
    'num_epochs': 100
}
logger.log_config(config)


# ============================================================================
# 3. TRAINING LOOP WITH LOGGING
# ============================================================================

best_val_dice = 0.0

for epoch in range(1, NUM_EPOCHS + 1):
    # Train
    train_loss, train_dice = train_one_epoch(train_loader, model, optimizer, loss_fn, device, epoch)
    
    # Validate
    val_loss, val_dice, val_iou, val_pa = validate_one_epoch(val_loader, model, loss_fn, device, epoch)
    
    # Update scheduler
    scheduler.step(val_dice)
    current_lr = optimizer.param_groups[0]['lr']
    
    # Log metrics
    logger.log_epoch(
        epoch=epoch,
        train_loss=train_loss,
        train_dice=train_dice,
        val_loss=val_loss,
        val_dice=val_dice,
        val_iou=val_iou,
        val_pixel_acc=val_pa,
        learning_rate=current_lr
    )
    
    # Save checkpoint
    is_best = val_dice > best_val_dice
    if is_best:
        best_val_dice = val_dice
    
    logger.save_checkpoint(
        epoch=epoch,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        val_dice=val_dice,
        is_best=is_best
    )
    
    # Generate plots periodically
    if epoch % 10 == 0:
        logger.plot_metrics()
    
    # Save prediction visualizations
    if epoch % 20 == 0:
        images, masks = next(iter(val_loader))
        images, masks = images.to(device), masks.to(device)
        
        with torch.no_grad():
            predictions = model(images)
        
        logger.save_predictions(
            images=images,
            masks_true=masks,
            masks_pred=predictions,
            epoch=epoch,
            num_samples=4
        )


# ============================================================================
# 4. FINAL REPORT AND PREDICTIONS
# ============================================================================

# Generate final report
logger.plot_metrics()
logger.generate_final_report()

# Load best model
best_checkpoint = torch.load(
    logger.checkpoint_dir / 'best_model.pth',
    map_location=device
)
model.load_state_dict(best_checkpoint['model_state_dict'])

# Save all predictions
from src.utils import PredictionSaver

pred_saver = PredictionSaver(
    output_dir='results/predictions',
    experiment_name='improved_unet_exp1'
)

pred_saver.save_from_dataloader(
    model=model,
    dataloader=test_loader,
    device=device,
    threshold=0.5
)


# ============================================================================
# 5. LOAD MODEL FROM CHECKPOINT
# ============================================================================

from src.utils import load_model_weights
from src.models import ImprovedUNet

model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
checkpoint = load_model_weights(
    model=model,
    filepath='results/checkpoints/my_exp_20251020_143022/best_model.pth',
    device=device
)

print(f"Loaded model from epoch {checkpoint['epoch']}")
print(f"Validation Dice: {checkpoint['val_dice']:.4f}")


# ============================================================================
# 6. REBUILD CACHE (IF NEEDED)
# ============================================================================

# If you changed image size or preprocessing logic
dataset = CachedHC18Dataset(
    image_dir='dataset/training_set/images',
    mask_dir='dataset/training_set/masks',
    cache_dir='preprocessed_data/train_cache',
    img_height=256,  # Changed from 512
    img_width=256,
    transform=transforms,
    force_rebuild=True  # Forces cache rebuild
)


# ============================================================================
# 7. SAVE INDIVIDUAL BATCH PREDICTIONS
# ============================================================================

from src.utils import PredictionSaver

saver = PredictionSaver(experiment_name='test_run')

for batch_idx, (images, masks) in enumerate(test_loader):
    images = images.to(device)
    
    with torch.no_grad():
        predictions = model(images)
    
    saver.save_batch(
        images=images,
        predictions=predictions,
        ground_truth=masks,
        filenames=None,  # Optional: provide list of filenames
        batch_idx=batch_idx,
        threshold=0.5
    )


# ============================================================================
# 8. ACCESS METRICS FROM CSV
# ============================================================================

import pandas as pd

# Read metrics
metrics = pd.read_csv('results/logs/improved_unet_exp1_20251020_143022/metrics.csv')

# Find best epoch
best_epoch = metrics.loc[metrics['val_dice'].idxmax()]
print(f"Best epoch: {best_epoch['epoch']}")
print(f"Best Dice: {best_epoch['val_dice']:.4f}")

# Plot custom metrics
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(metrics['epoch'], metrics['val_dice'], label='Validation Dice')
plt.axhline(y=0.9781, color='r', linestyle='--', label='Target (97.81%)')
plt.xlabel('Epoch')
plt.ylabel('Dice Score')
plt.legend()
plt.grid(True)
plt.savefig('custom_dice_plot.png')


# ============================================================================
# 9. COMPLETE MINIMAL EXAMPLE
# ============================================================================

import torch
from torch.utils.data import DataLoader
from src.data import CachedHC18Dataset
from src.models import ImprovedUNet
from src.losses import DiceBCELoss
from src.utils import get_transforms, TrainingLogger

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger = TrainingLogger(experiment_name='quick_test')

# Data
train_dataset = CachedHC18Dataset(
    'dataset/training_set/images',
    'dataset/training_set/masks',
    'preprocessed_data/train_cache',
    transform=get_transforms(256, 256, is_train=True)
)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Model
model = ImprovedUNet(in_channels=1, out_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
loss_fn = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)

# Train
for epoch in range(1, 11):
    model.train()
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)
        
        predictions = model(images)
        loss = loss_fn(predictions, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

logger.generate_final_report()
