"""
Example: Training with Cached Dataset and Logger

This script demonstrates how to use:
1. CachedHC18Dataset for faster data loading
2. TrainingLogger for comprehensive metric tracking
3. PredictionSaver for saving model outputs
4. Configuration from YAML file

Usage:
    python examples/train_with_caching.py [--config path/to/config.yaml]
"""

import sys
import os
import argparse
from pathlib import Path

# Add parent directory to path to import src module
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from configs.config_loader import load_config
from src.data import CachedHC18Dataset
from src.models import ImprovedUNet
from src.losses import DiceBCELoss
from src.utils import get_transforms, TrainingLogger, PredictionSaver
from src.utils.train import train_one_epoch, validate_one_epoch


def main():
    """Main training loop with caching and logging."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train with caching and logging')
    parser.add_argument('--config', type=str, default='configs/train_config.yaml',
                       help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Extract config values
    data_cfg = config['data']
    model_cfg = config['model']
    train_cfg = config['training']
    aug_cfg = config['augmentation']
    checkpoint_cfg = config['checkpoint']
    logging_cfg = config['logging']
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get image size from config
    img_size = aug_cfg['preprocessing']['image_size'][0]
    
    print("="*70)
    print("TRAINING WITH CACHED DATASET AND LOGGING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch Size: {train_cfg['batch_size']}")
    print(f"Image Size: {img_size}x{img_size}")
    print(f"Use Cache: {data_cfg['use_cache']}")
    print("="*70)
    
    # ========================================================================
    # 1. INITIALIZE LOGGER
    # ========================================================================
    
    experiment_name = f"improved_unet_{data_cfg.get('use_cache', 'cached')}"
    
    logger = TrainingLogger(
        experiment_name=experiment_name,
        base_dir=logging_cfg['log_dir'],
        save_plots=True
    )
    
    # Save configuration (log the full config dict)
    logger.log_config(config)
    
    # ========================================================================
    # 2. CREATE CACHED DATASETS
    # ========================================================================
    
    print("\n[1/5] Creating cached datasets...")
    
    # Get transforms (augmentation only, preprocessing is cached)
    train_transforms = get_transforms(img_size, img_size, is_train=True)
    val_transforms = get_transforms(img_size, img_size, is_train=False)
    
    # Create cached datasets (will build cache on first run)
    train_dataset = CachedHC18Dataset(
        image_dir=data_cfg['train_images'],
        mask_dir=data_cfg['train_masks'],
        cache_dir=f"{data_cfg['cache_dir']}/train_cache",
        img_height=img_size,
        img_width=img_size,
        transform=train_transforms,
        force_rebuild=False  # Set to True to rebuild cache
    )
    
    val_dataset = CachedHC18Dataset(
        image_dir=data_cfg['val_images'],
        mask_dir=data_cfg['val_masks'],
        cache_dir=f"{data_cfg['cache_dir']}/val_cache",
        img_height=img_size,
        img_width=img_size,
        transform=val_transforms,
        force_rebuild=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg['num_workers'],
        pin_memory=train_cfg['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg['num_workers'],
        pin_memory=train_cfg['pin_memory']
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # ========================================================================
    # 3. INITIALIZE MODEL, LOSS, OPTIMIZER
    # ========================================================================
    
    print("\n[2/5] Initializing model...")
    
    model = ImprovedUNet(
        in_channels=model_cfg['in_channels'],
        out_channels=model_cfg['out_channels']
    ).to(device)
    
    # Loss function with configurable weights
    loss_cfg = train_cfg['loss']
    loss_fn = DiceBCELoss(
        dice_weight=loss_cfg['dice_weight'],
        bce_weight=loss_cfg['bce_weight']
    )
    
    optimizer_cfg = train_cfg['optimizer']
    optimizer = Adam(
        model.parameters(),
        lr=optimizer_cfg['lr'],
        betas=tuple(optimizer_cfg['betas']),
        eps=optimizer_cfg['eps'],
        weight_decay=optimizer_cfg['weight_decay']
    )
    
    scheduler_cfg = train_cfg['scheduler']
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg['mode'],
        factor=scheduler_cfg['factor'],
        patience=scheduler_cfg['patience']
    )
    
    # ========================================================================
    # 4. TRAINING LOOP
    # ========================================================================
    
    print("\n[3/5] Starting training...")
    
    best_val_dice = 0.0
    
    for epoch in range(1, train_cfg['num_epochs'] + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{train_cfg['num_epochs']}")
        print(f"{'='*70}")
        
        # Train
        train_loss, train_dice = train_one_epoch(
            loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch
        )
        
        # Validate
        val_loss, val_dice, val_iou, val_pa = validate_one_epoch(
            loader=val_loader,
            model=model,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch
        )
        
        # Update learning rate
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
        
        # Plot metrics every 10 epochs
        if epoch % 10 == 0:
            logger.plot_metrics()
        
        # Save predictions every 20 epochs
        if epoch % 20 == 0:
            # Get a batch from validation set
            images, masks = next(iter(val_loader))
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.no_grad():
                predictions = model(images)
            
            logger.save_predictions(
                images=images,
                masks_true=masks,
                masks_pred=predictions,
                epoch=epoch,
                num_samples=4
            )
    
    # ========================================================================
    # 5. FINAL EVALUATION AND PREDICTIONS
    # ========================================================================
    
    print("\n[4/5] Generating final predictions...")
    
    # Load best model
    best_checkpoint = torch.load(
        logger.checkpoint_dir / 'best_model.pth',
        map_location=device
    )
    model.load_state_dict(best_checkpoint['model_state_dict'])
    
    # Save all predictions on validation set
    pred_saver = PredictionSaver(
        output_dir=logging_cfg.get('prediction_dir', 'results/predictions'),
        experiment_name=experiment_name
    )
    
    pred_saver.save_from_dataloader(
        model=model,
        dataloader=val_loader,
        device=device,
        threshold=0.5
    )
    
    # ========================================================================
    # 6. GENERATE FINAL REPORT
    # ========================================================================
    
    print("\n[5/5] Generating final report...")
    logger.plot_metrics()
    logger.generate_final_report()
    
    print("\n✓ Training complete!")
    print(f"Best Validation Dice: {best_val_dice:.4f}")
    
    # Print target metrics comparison
    target_metrics = config.get('target_metrics', {})
    if target_metrics:
        print("\nTarget Performance Metrics:")
        print(f"  Target Dice: {target_metrics.get('dice', 0)*100:.2f}% | Achieved: {best_val_dice*100:.2f}%")


if __name__ == '__main__':
    main()
