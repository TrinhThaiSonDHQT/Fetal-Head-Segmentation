"""
Training Script for Standard U-Net Model
Fetal Head Segmentation in Ultrasound Images
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.standard_unet import StandardUNet
from src.data.cached_dataset import CachedDataset
from src.metrics.segmentation_metrics import dice_coefficient, iou_score, pixel_accuracy
from src.utils.visualization import save_prediction_grid
from src.utils.device_utils import get_device, print_device_info


class StandardUNetTrainer:
    """
    Trainer class for Standard U-Net model
    """
    def __init__(self, config_path='configs/standard_unet_config.yaml'):
        """
        Initialize trainer with configuration
        
        Args:
            config_path (str): Path to YAML configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set device (automatically handles CPU/GPU)
        prefer_gpu = self.config.get('device', 'cuda') == 'cuda'
        self.device = get_device(prefer_gpu=prefer_gpu)
        
        # Store device in config for consistency
        self.config['device'] = str(self.device)
        
        # Create directories
        self._create_directories()
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize loss function (Binary Cross-Entropy)
        self.criterion = nn.BCELoss()
        
        # Initialize optimizer (Adam with lr=1e-4)
        self.optimizer = self._build_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._build_scheduler()
        
        # Initialize dataloaders
        self.train_loader, self.val_loader = self._build_dataloaders()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_pa': [],
            'lr': []
        }
        
        self.best_dice = 0.0
        self.epochs_without_improvement = 0
        
    def _create_directories(self):
        """Create necessary directories for logging and checkpoints"""
        dirs = [
            self.config['logging']['checkpoint_dir'],
            self.config['logging']['log_dir'],
            self.config['logging']['prediction_dir'],
            self.config['logging']['visualization_dir']
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
    
    def _build_model(self):
        """Build Standard U-Net model"""
        model = StandardUNet(
            in_channels=self.config['model']['in_channels'],
            out_channels=self.config['model']['out_channels'],
            base_filters=self.config['model']['base_filters']
        )
        model = model.to(self.device)
        
        total_params = model.count_parameters()
        print(f"\nStandard U-Net Model:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Base filters: {self.config['model']['base_filters']}")
        
        return model
    
    def _build_optimizer(self):
        """Build Adam optimizer with lr=1e-4"""
        optimizer_config = self.config['training']['optimizer']
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=optimizer_config['lr'],  # 1e-4
            betas=tuple(optimizer_config['betas']),
            eps=optimizer_config['eps'],
            weight_decay=optimizer_config['weight_decay']
        )
        print(f"\nOptimizer: Adam")
        print(f"  Learning rate: {optimizer_config['lr']}")
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        scheduler_config = self.config['training']['scheduler']
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=scheduler_config['mode'],
            factor=scheduler_config['factor'],
            patience=scheduler_config['patience'],
            min_lr=scheduler_config['min_lr'],
            verbose=scheduler_config['verbose']
        )
        return scheduler
    
    def _build_dataloaders(self):
        """Build train and validation dataloaders with normalization"""
        data_config = self.config['data']
        training_config = self.config['training']
        
        # Train dataset
        train_dataset = CachedDataset(
            image_dir=data_config['train_images'],
            mask_dir=data_config['train_masks'],
            cache_dir=os.path.join(data_config['cache_dir'], 'train_cache'),
            split='train',
            use_cache=data_config['use_cache']
        )
        
        # Validation dataset
        val_dataset = CachedDataset(
            image_dir=data_config['val_images'],
            mask_dir=data_config['val_masks'],
            cache_dir=os.path.join(data_config['cache_dir'], 'val_cache'),
            split='val',
            use_cache=data_config['use_cache']
        )
        
        # DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=training_config['batch_size'],
            shuffle=True,
            num_workers=training_config['num_workers'],
            pin_memory=training_config['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=training_config['batch_size'],
            shuffle=False,
            num_workers=training_config['num_workers'],
            pin_memory=training_config['pin_memory']
        )
        
        print(f"\nDatasets:")
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Validation samples: {len(val_dataset)}")
        print(f"  Batch size: {training_config['batch_size']}")
        print(f"  Normalization: Divide by 255.0")
        
        return train_loader, val_loader
    
    def train_one_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # Calculate loss (Binary Cross-Entropy)
            loss = self.criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        epoch_loss = running_loss / len(self.train_loader)
        return epoch_loss
    
    def validate(self, epoch):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        dice_scores = []
        iou_scores = []
        pa_scores = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch_idx, (images, masks) in enumerate(pbar):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Calculate loss
                loss = self.criterion(outputs, masks)
                running_loss += loss.item()
                
                # Calculate metrics
                preds = (outputs > 0.5).float()
                
                for i in range(images.size(0)):
                    dice = dice_coefficient(preds[i], masks[i])
                    iou = iou_score(preds[i], masks[i])
                    pa = pixel_accuracy(preds[i], masks[i])
                    
                    dice_scores.append(dice.item())
                    iou_scores.append(iou.item())
                    pa_scores.append(pa.item())
                
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'dice': f"{np.mean(dice_scores):.4f}"
                })
        
        # Calculate average metrics
        val_loss = running_loss / len(self.val_loader)
        val_dice = np.mean(dice_scores)
        val_iou = np.mean(iou_scores)
        val_pa = np.mean(pa_scores)
        
        return val_loss, val_dice, val_iou, val_pa
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_dice': self.best_dice,
            'history': self.history,
            'config': self.config
        }
        
        checkpoint_dir = self.config['logging']['checkpoint_dir']
        
        # Save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  → Saved best model (Dice: {self.best_dice:.4f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % self.config['logging']['save_every_n_epochs'] == 0:
            epoch_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, epoch_path)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss (BCE)')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice coefficient
        axes[0, 1].plot(self.history['val_dice'], label='Val Dice', color='green')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Coefficient')
        axes[0, 1].set_title('Validation Dice Coefficient')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # IoU
        axes[1, 0].plot(self.history['val_iou'], label='Val IoU', color='orange')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('IoU Score')
        axes[1, 0].set_title('Validation IoU Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['lr'], label='Learning Rate', color='red')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        save_path = os.path.join(self.config['logging']['log_dir'], 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def train(self):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        patience = self.config['training']['early_stopping_patience']
        
        print(f"\n{'='*60}")
        print(f"Starting Training - Standard U-Net")
        print(f"{'='*60}")
        print(f"Epochs: {num_epochs}")
        print(f"Loss Function: Binary Cross-Entropy")
        print(f"Early Stopping Patience: {patience}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_one_epoch(epoch)
            
            # Validate
            val_loss, val_dice, val_iou, val_pa = self.validate(epoch)
            
            # Update learning rate
            self.scheduler.step(val_dice)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_dice'].append(val_dice)
            self.history['val_iou'].append(val_iou)
            self.history['val_pa'].append(val_pa)
            self.history['lr'].append(current_lr)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Dice:   {val_dice:.4f}")
            print(f"  Val IoU:    {val_iou:.4f}")
            print(f"  Val PA:     {val_pa:.4f}")
            print(f"  LR:         {current_lr:.6f}")
            
            # Check for improvement
            is_best = val_dice > self.best_dice
            if is_best:
                self.best_dice = val_dice
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, is_best)
            
            # Plot training curves
            if (epoch + 1) % self.config['logging']['visualize_every_n_epochs'] == 0:
                self.plot_training_curves()
            
            # Early stopping
            if self.epochs_without_improvement >= patience:
                print(f"\n{'='*60}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best Dice Score: {self.best_dice:.4f}")
                print(f"{'='*60}")
                break
        
        # Final summary
        print(f"\n{'='*60}")
        print(f"Training Completed!")
        print(f"{'='*60}")
        print(f"Best Validation Dice: {self.best_dice:.4f}")
        print(f"Best Validation IoU:  {max(self.history['val_iou']):.4f}")
        print(f"Best Validation PA:   {max(self.history['val_pa']):.4f}")
        print(f"{'='*60}\n")
        
        # Save final training curves
        self.plot_training_curves()


def main():
    """Main function"""
    # Print device information
    print_device_info()
    print()
    
    # Create trainer
    trainer = StandardUNetTrainer(config_path='configs/standard_unet_config.yaml')
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
