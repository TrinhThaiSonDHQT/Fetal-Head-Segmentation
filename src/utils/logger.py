"""
Training Logger for Metrics and Visualizations

Handles logging of training metrics, saving checkpoints, and creating visualizations.
"""
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import torch
import matplotlib.pyplot as plt
import numpy as np


class TrainingLogger:
    """
    Comprehensive logger for training metrics and outputs.
    
    Organizes logs, checkpoints, and visualizations in a structured manner.
    
    Args:
        experiment_name (str): Name of the experiment/run
        base_dir (str): Base directory for all results (default: 'results')
        save_plots (bool): Whether to save plots during training
    """
    
    def __init__(
        self,
        experiment_name: str,
        base_dir: str = 'results',
        save_plots: bool = True
    ):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.save_plots = save_plots
        
        # Create timestamp for unique run identification
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"{experiment_name}_{self.timestamp}"
        
        # Define output directories
        self.log_dir = self.base_dir / 'logs' / self.run_name
        self.checkpoint_dir = self.base_dir / 'checkpoints' / self.run_name
        self.viz_dir = self.base_dir / 'visualizations' / self.run_name
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metric storage
        self.metrics_history: Dict[str, List[float]] = {
            'epoch': [],
            'train_loss': [],
            'train_dice': [],
            'val_loss': [],
            'val_dice': [],
            'val_iou': [],
            'val_pixel_acc': [],
            'learning_rate': []
        }
        
        # CSV file for metrics
        self.csv_file = self.log_dir / 'metrics.csv'
        self._init_csv()
        
        # JSON file for configuration
        self.config_file = self.log_dir / 'config.json'
        
        print(f"Logger initialized: {self.run_name}")
        print(f"  Logs: {self.log_dir}")
        print(f"  Checkpoints: {self.checkpoint_dir}")
        print(f"  Visualizations: {self.viz_dir}")
    
    def _init_csv(self):
        """Initialize CSV file with headers."""
        with open(self.csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_dice', 
                'val_loss', 'val_dice', 'val_iou', 'val_pixel_acc', 
                'learning_rate'
            ])
    
    def log_config(self, config: Dict):
        """
        Save training configuration to JSON.
        
        Args:
            config (dict): Configuration dictionary with hyperparameters
        """
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Configuration saved to {self.config_file}")
    
    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_dice: float,
        val_loss: float,
        val_dice: float,
        val_iou: float,
        val_pixel_acc: float,
        learning_rate: float
    ):
        """
        Log metrics for a single epoch.
        
        Args:
            epoch (int): Current epoch number
            train_loss (float): Training loss
            train_dice (float): Training Dice score
            val_loss (float): Validation loss
            val_dice (float): Validation Dice score
            val_iou (float): Validation IoU
            val_pixel_acc (float): Validation pixel accuracy
            learning_rate (float): Current learning rate
        """
        # Store in history
        self.metrics_history['epoch'].append(epoch)
        self.metrics_history['train_loss'].append(train_loss)
        self.metrics_history['train_dice'].append(train_dice)
        self.metrics_history['val_loss'].append(val_loss)
        self.metrics_history['val_dice'].append(val_dice)
        self.metrics_history['val_iou'].append(val_iou)
        self.metrics_history['val_pixel_acc'].append(val_pixel_acc)
        self.metrics_history['learning_rate'].append(learning_rate)
        
        # Append to CSV
        with open(self.csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, train_loss, train_dice,
                val_loss, val_dice, val_iou, val_pixel_acc,
                learning_rate
            ])
        
        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, "
              f"IoU: {val_iou:.4f}, PA: {val_pixel_acc:.4f}")
        print(f"  LR: {learning_rate:.6f}")
    
    def save_checkpoint(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        val_dice: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint.
        
        Args:
            epoch (int): Current epoch
            model (nn.Module): Model to save
            optimizer (Optimizer): Optimizer state
            scheduler (optional): Learning rate scheduler
            val_dice (float): Validation Dice score
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_dice': val_dice,
            'metrics_history': self.metrics_history
        }
        
        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model separately
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"  ✓ Best model saved (Dice: {val_dice:.4f})")
    
    def plot_metrics(self):
        """Generate and save training metric plots."""
        if not self.save_plots or len(self.metrics_history['epoch']) < 2:
            return
        
        epochs = self.metrics_history['epoch']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Metrics - {self.experiment_name}', fontsize=16)
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        ax.plot(epochs, self.metrics_history['train_loss'], label='Train Loss', marker='o')
        ax.plot(epochs, self.metrics_history['val_loss'], label='Val Loss', marker='s')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True)
        
        # Plot 2: Dice score
        ax = axes[0, 1]
        ax.plot(epochs, self.metrics_history['train_dice'], label='Train Dice', marker='o')
        ax.plot(epochs, self.metrics_history['val_dice'], label='Val Dice', marker='s')
        ax.axhline(y=0.9781, color='r', linestyle='--', label='Target (97.81%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dice Score')
        ax.set_title('Dice Similarity Coefficient')
        ax.legend()
        ax.grid(True)
        
        # Plot 3: IoU and Pixel Accuracy
        ax = axes[1, 0]
        ax.plot(epochs, self.metrics_history['val_iou'], label='Val IoU', marker='o', color='green')
        ax.axhline(y=0.9790, color='r', linestyle='--', label='Target IoU (97.90%)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('IoU')
        ax.set_title('Mean Intersection over Union')
        ax.legend()
        ax.grid(True)
        
        # Plot 4: Learning rate
        ax = axes[1, 1]
        ax.plot(epochs, self.metrics_history['learning_rate'], marker='o', color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        plot_path = self.viz_dir / 'training_metrics.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Metrics plot saved to {plot_path}")
    
    def save_predictions(
        self,
        images: torch.Tensor,
        masks_true: torch.Tensor,
        masks_pred: torch.Tensor,
        epoch: int,
        num_samples: int = 4
    ):
        """
        Save visualization of predictions vs ground truth.
        
        Args:
            images (Tensor): Input images (B, 1, H, W)
            masks_true (Tensor): Ground truth masks (B, 1, H, W)
            masks_pred (Tensor): Predicted masks (B, 1, H, W)
            epoch (int): Current epoch
            num_samples (int): Number of samples to visualize
        """
        if not self.save_plots:
            return
        
        num_samples = min(num_samples, images.size(0))
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]
        
        for i in range(num_samples):
            # Convert to numpy
            img = images[i, 0].cpu().numpy()
            mask_true = masks_true[i, 0].cpu().numpy()
            mask_pred = masks_pred[i, 0].cpu().numpy()
            
            # Plot image
            axes[i, 0].imshow(img, cmap='gray')
            axes[i, 0].set_title('Input Image')
            axes[i, 0].axis('off')
            
            # Plot ground truth
            axes[i, 1].imshow(img, cmap='gray')
            axes[i, 1].imshow(mask_true, cmap='Reds', alpha=0.5)
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            # Plot prediction
            axes[i, 2].imshow(img, cmap='gray')
            axes[i, 2].imshow(mask_pred, cmap='Blues', alpha=0.5)
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        pred_path = self.viz_dir / f'predictions_epoch_{epoch:03d}.png'
        plt.savefig(pred_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Predictions saved to {pred_path}")
    
    def generate_final_report(self):
        """Generate final training report with summary statistics."""
        report_path = self.log_dir / 'training_report.txt'
        
        # Get best metrics
        best_epoch = np.argmax(self.metrics_history['val_dice'])
        best_dice = self.metrics_history['val_dice'][best_epoch]
        best_iou = self.metrics_history['val_iou'][best_epoch]
        best_pa = self.metrics_history['val_pixel_acc'][best_epoch]
        
        # Generate report
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TRAINING REPORT - IMPROVED U-NET\n")
            f.write("="*70 + "\n\n")
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"Run ID: {self.run_name}\n")
            f.write(f"Timestamp: {self.timestamp}\n\n")
            
            f.write("BEST PERFORMANCE:\n")
            f.write("-"*70 + "\n")
            f.write(f"Best Epoch: {self.metrics_history['epoch'][best_epoch]}\n")
            f.write(f"Validation Dice Score: {best_dice:.4f} ({best_dice*100:.2f}%)\n")
            f.write(f"Validation IoU: {best_iou:.4f} ({best_iou*100:.2f}%)\n")
            f.write(f"Validation Pixel Accuracy: {best_pa:.4f} ({best_pa*100:.2f}%)\n\n")
            
            f.write("TARGET PERFORMANCE METRICS:\n")
            f.write("-"*70 + "\n")
            f.write(f"Target Dice Score: 97.81% {'✓ ACHIEVED' if best_dice >= 0.9781 else '✗ NOT ACHIEVED'}\n")
            f.write(f"Target IoU: 97.90% {'✓ ACHIEVED' if best_iou >= 0.9790 else '✗ NOT ACHIEVED'}\n")
            f.write(f"Target Pixel Accuracy: 99.18% {'✓ ACHIEVED' if best_pa >= 0.9918 else '✗ NOT ACHIEVED'}\n\n")
            
            f.write("FILES:\n")
            f.write("-"*70 + "\n")
            f.write(f"Metrics CSV: {self.csv_file}\n")
            f.write(f"Best Model: {self.checkpoint_dir / 'best_model.pth'}\n")
            f.write(f"Latest Checkpoint: {self.checkpoint_dir / 'latest_checkpoint.pth'}\n")
            f.write(f"Training Plots: {self.viz_dir / 'training_metrics.png'}\n")
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETE - Report saved to {report_path}")
        print(f"{'='*70}")
        print(f"Best Validation Dice: {best_dice:.4f} ({best_dice*100:.2f}%)")
        print(f"Best Validation IoU: {best_iou:.4f} ({best_iou*100:.2f}%)")
        print(f"Best Validation PA: {best_pa:.4f} ({best_pa*100:.2f}%)")
        print(f"{'='*70}\n")
