"""
Main Training Script for Fetal Head Segmentation using Improved U-Net

This script orchestrates the entire training and evaluation process.

Target Performance Metrics:
- DSC (Dice Similarity Coefficient): ≥97.81%
- mIoU (Mean Intersection over Union): ≥97.90%
- mPA (Mean Pixel Accuracy): ≥99.18%

Usage:
    python main.py [--config path/to/config.yaml]
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
import os
import argparse
from pathlib import Path

from configs.config_loader import load_config
from src.data import HC18Dataset, CachedHC18Dataset
from src.models import ImprovedUNet
from src.losses import DiceBCELoss
from src.utils import get_transforms
from src.utils.train import train_one_epoch, evaluate_model


# ============================================================================
# MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training and evaluation pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Improved U-Net for Fetal Head Segmentation')
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
    
    # Create output directories
    os.makedirs(checkpoint_cfg['save_dir'], exist_ok=True)
    os.makedirs(logging_cfg['log_dir'], exist_ok=True)
    if logging_cfg.get('save_predictions', False):
        os.makedirs(logging_cfg['prediction_dir'], exist_ok=True)
        os.makedirs(logging_cfg['visualization_dir'], exist_ok=True)
    
    print("="*70)
    print("FETAL HEAD SEGMENTATION - IMPROVED U-NET TRAINING")
    print("="*70)
    print(f"Device: {device}")
    print(f"Batch Size: {train_cfg['batch_size']}")
    print(f"Learning Rate: {train_cfg['optimizer']['lr']}")
    print(f"Number of Epochs: {train_cfg['num_epochs']}")
    print("="*70)
    
    # ========================================================================
    # 1. DATA LOADING
    # ========================================================================
    
    print("\n[1/4] Loading datasets...")
    
    # Get image size from config
    img_size = aug_cfg['preprocessing']['image_size'][0]  # Assuming square images
    
    # Get transforms
    train_transforms = get_transforms(img_size, img_size, is_train=True)
    val_transforms = get_transforms(img_size, img_size, is_train=False)
    
    # Determine whether to use cached dataset
    use_cache = data_cfg.get('use_cache', False)
    cache_dir = data_cfg.get('cache_dir', 'preprocessed_data')
    train_cache_path = Path(cache_dir) / 'train_cache'
    val_cache_path = Path(cache_dir) / 'val_cache'
    
    # Check if cache directories exist and have files
    train_cache_exists = train_cache_path.exists() and len(list(train_cache_path.glob('*.npz'))) > 0
    val_cache_exists = val_cache_path.exists() and len(list(val_cache_path.glob('*.npz'))) > 0
    
    # Create datasets based on cache availability
    if use_cache and train_cache_exists:
        print("Using cached training dataset...")
        train_dataset = CachedHC18Dataset(
            image_dir=data_cfg['train_images'],
            mask_dir=data_cfg['train_masks'],
            cache_dir=str(train_cache_path),
            img_height=img_size,
            img_width=img_size,
            transform=train_transforms,
            force_rebuild=False
        )
    else:
        if use_cache and not train_cache_exists:
            print("Cache not found. Using standard dataset for training...")
        train_dataset = HC18Dataset(
            data_cfg['train_images'], 
            data_cfg['train_masks'], 
            transform=train_transforms
        )
    
    if use_cache and val_cache_exists:
        print("Using cached validation dataset...")
        val_dataset = CachedHC18Dataset(
            image_dir=data_cfg['val_images'],
            mask_dir=data_cfg['val_masks'],
            cache_dir=str(val_cache_path),
            img_height=img_size,
            img_width=img_size,
            transform=val_transforms,
            force_rebuild=False
        )
    else:
        if use_cache and not val_cache_exists:
            print("Cache not found. Using standard dataset for validation...")
        val_dataset = HC18Dataset(
            data_cfg['val_images'], 
            data_cfg['val_masks'], 
            transform=val_transforms
        )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples available at: {data_cfg['test_images']}")
    
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
    
    # ========================================================================
    # 2. MODEL INITIALIZATION
    # ========================================================================
    
    print("\n[2/4] Initializing model...")
    
    # Initialize Improved U-Net model
    model = ImprovedUNet(
        in_channels=model_cfg['in_channels'], 
        out_channels=model_cfg['out_channels']
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function with configurable weights
    loss_cfg = train_cfg['loss']
    loss_fn = DiceBCELoss(
        dice_weight=loss_cfg['dice_weight'],
        bce_weight=loss_cfg['bce_weight']
    )
    
    # Optimizer
    optimizer_cfg = train_cfg['optimizer']
    optimizer = Adam(
        model.parameters(), 
        lr=optimizer_cfg['lr'],
        betas=tuple(optimizer_cfg['betas']),
        eps=optimizer_cfg['eps'],
        weight_decay=optimizer_cfg['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler_cfg = train_cfg['scheduler']
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode=scheduler_cfg['mode'],
        factor=scheduler_cfg['factor'],
        patience=scheduler_cfg['patience'],
        min_lr=scheduler_cfg['min_lr'],
        verbose=scheduler_cfg['verbose']
    )
    
    print("Model initialization complete.")
    
    # ========================================================================
    # 3. TRAINING LOOP
    # ========================================================================
    
    print("\n[3/4] Starting training...")
    print("="*70)
    
    best_dice = 0.0
    # monitor_metric = checkpoint_cfg['monitor']  # 'val_dice'
    
    for epoch in range(1, train_cfg['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{train_cfg['num_epochs']}")
        print("-" * 70)
        
        # Train for one epoch
        train_loss, train_dice = train_one_epoch(
            train_loader, model, optimizer, loss_fn, device, epoch
        )
        
        # Evaluate on validation set
        val_metrics = evaluate_model(val_loader, model, loss_fn, device)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Dice: {val_metrics['dice']:.4f}")
        print(f"Val mIoU: {val_metrics['miou']:.4f} | Val mPA: {val_metrics['pixel_accuracy']:.4f}")
        
        # Update learning rate based on validation Dice
        scheduler.step(val_metrics['dice'])
        
        # Save best model
        if checkpoint_cfg['save_best'] and val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            save_path = os.path.join(checkpoint_cfg['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_dice': best_dice,
                'val_metrics': val_metrics,
                'config': config
            }, save_path)
            print(f"✓ Saved best model with Dice: {best_dice:.4f}")
        
        # Save last checkpoint
        if checkpoint_cfg['save_last']:
            save_path = os.path.join(checkpoint_cfg['save_dir'], 'last_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, save_path)
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(checkpoint_cfg['save_dir'], f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, save_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")
    
    # ========================================================================
    # 4. TRAINING COMPLETE
    # ========================================================================
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED!")
    print("="*70)
    print(f"Best Dice Score: {best_dice:.4f}")
    print(f"Best model saved at: {os.path.join(checkpoint_cfg['save_dir'], 'best_model.pth')}")
    
    # Print target metrics comparison
    target_metrics = config.get('target_metrics', {})
    if target_metrics:
        print("\nTarget Performance Metrics:")
        print(f"  Target Dice: {target_metrics.get('dice', 0)*100:.2f}% | Achieved: {best_dice*100:.2f}%")
    
    print("="*70)


if __name__ == "__main__":
    main()
