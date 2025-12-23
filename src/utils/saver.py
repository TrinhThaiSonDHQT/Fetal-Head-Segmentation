"""
Prediction Saver for Test Set Evaluation

Saves model predictions on test set with various output formats.
"""
import os
from pathlib import Path
from typing import Optional
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class PredictionSaver:
    """
    Save model predictions for test/validation datasets.
    
    Saves predictions in multiple formats:
    - Binary masks (PNG)
    - Probability maps (PNG)
    - Visualizations with overlays (PNG)
    
    Args:
        output_dir (str): Base directory for saving predictions
        experiment_name (str): Name of the experiment
    """
    
    def __init__(self, output_dir: str = 'results/predictions', experiment_name: str = 'test'):
        self.output_dir = Path(output_dir) / experiment_name
        
        # Create subdirectories
        self.masks_dir = self.output_dir / 'masks'
        self.probs_dir = self.output_dir / 'probabilities'
        self.overlay_dir = self.output_dir / 'overlays'
        
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.probs_dir.mkdir(parents=True, exist_ok=True)
        self.overlay_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"PredictionSaver initialized:")
        print(f"  Masks: {self.masks_dir}")
        print(f"  Probabilities: {self.probs_dir}")
        print(f"  Overlays: {self.overlay_dir}")
    
    def save_batch(
        self,
        images: torch.Tensor,
        predictions: torch.Tensor,
        ground_truth: Optional[torch.Tensor] = None,
        filenames: Optional[list] = None,
        batch_idx: int = 0,
        threshold: float = 0.5
    ):
        """
        Save predictions for a batch of images.
        
        Args:
            images (Tensor): Input images (B, 1, H, W)
            predictions (Tensor): Model predictions (B, 1, H, W), values in [0, 1]
            ground_truth (Tensor, optional): Ground truth masks (B, 1, H, W)
            filenames (list, optional): List of filenames for each image
            batch_idx (int): Batch index for naming
            threshold (float): Threshold for converting probabilities to binary masks
        """
        batch_size = images.size(0)
        
        for i in range(batch_size):
            # Generate filename
            if filenames and i < len(filenames):
                filename = Path(filenames[i]).stem
            else:
                filename = f"batch{batch_idx:04d}_img{i:02d}"
            
            # Extract tensors
            img = images[i, 0].cpu().numpy()  # (H, W)
            pred = predictions[i, 0].cpu().numpy()  # (H, W)
            gt = ground_truth[i, 0].cpu().numpy() if ground_truth is not None else None
            
            # 1. Save binary mask
            binary_mask = (pred > threshold).astype(np.uint8) * 255
            mask_path = self.masks_dir / f"{filename}_mask.png"
            cv2.imwrite(str(mask_path), binary_mask)
            
            # 2. Save probability map
            prob_map = (pred * 255).astype(np.uint8)
            prob_path = self.probs_dir / f"{filename}_prob.png"
            cv2.imwrite(str(prob_path), prob_map)
            
            # 3. Save overlay visualization
            self._save_overlay(img, pred, gt, filename)
    
    def _save_overlay(
        self,
        image: np.ndarray,
        prediction: np.ndarray,
        ground_truth: Optional[np.ndarray],
        filename: str
    ):
        """Create and save overlay visualization."""
        # Determine number of subplots
        num_plots = 3 if ground_truth is not None else 2
        
        fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 5))
        if num_plots == 2:
            axes = [axes[0], axes[1]]
        
        # Normalize image to [0, 1] for display
        img_display = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Plot 1: Original image
        axes[0].imshow(img_display, cmap='gray')
        axes[0].set_title('Input Image')
        axes[0].axis('off')
        
        # Plot 2: Prediction overlay
        axes[1].imshow(img_display, cmap='gray')
        axes[1].imshow(prediction, cmap='Blues', alpha=0.5)
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Plot 3: Ground truth overlay (if available)
        if ground_truth is not None:
            axes[2].imshow(img_display, cmap='gray')
            axes[2].imshow(ground_truth, cmap='Reds', alpha=0.5)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save
        overlay_path = self.overlay_dir / f"{filename}_overlay.png"
        plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def save_from_dataloader(
        self,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        threshold: float = 0.5
    ):
        """
        Run inference on entire dataloader and save all predictions.
        
        Args:
            model (nn.Module): Trained model
            dataloader (DataLoader): DataLoader for test/validation set
            device (torch.device): Device to run inference on
            threshold (float): Threshold for binary masks
        """
        model.eval()
        
        print(f"Generating predictions for {len(dataloader)} batches...")
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(tqdm(dataloader, desc="Saving predictions")):
                images = images.to(device)
                masks = masks.to(device)
                
                # Forward pass
                predictions = model(images)
                
                # Save batch
                self.save_batch(
                    images=images,
                    predictions=predictions,
                    ground_truth=masks,
                    batch_idx=batch_idx,
                    threshold=threshold
                )
        
        print(f"✓ All predictions saved to {self.output_dir}")


def save_model_weights(
    model: torch.nn.Module,
    filepath: str,
    metadata: Optional[dict] = None
):
    """
    Save model weights with optional metadata.
    
    Args:
        model (nn.Module): Model to save
        filepath (str): Path to save the model
        metadata (dict, optional): Additional metadata (epoch, metrics, etc.)
    """
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_architecture': str(model.__class__.__name__),
    }
    
    if metadata:
        save_dict.update(metadata)
    
    torch.save(save_dict, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model_weights(
    model: torch.nn.Module,
    filepath: str,
    device: torch.device
) -> dict:
    """
    Load model weights from checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into
        filepath (str): Path to checkpoint file
        device (torch.device): Device to load model on
    
    Returns:
        dict: Checkpoint metadata
    """
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Model weights loaded from {filepath}")
    
    return checkpoint
