"""
Visualization Utilities for Segmentation Results
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_prediction_grid(images, masks, predictions, save_path, num_samples=4):
    """
    Save a grid visualization of images, ground truth masks, and predictions
    
    Args:
        images (torch.Tensor): Input images [B, C, H, W]
        masks (torch.Tensor): Ground truth masks [B, C, H, W]
        predictions (torch.Tensor): Predicted masks [B, C, H, W]
        save_path (str): Path to save the visualization
        num_samples (int): Number of samples to visualize
    """
    # Convert to numpy and move to CPU
    images = images.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    predictions = predictions.detach().cpu().numpy()
    
    # Limit number of samples
    num_samples = min(num_samples, images.shape[0])
    
    # Create figure
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Image
        axes[i, 0].imshow(images[i, 0], cmap='gray')
        axes[i, 0].set_title('Input Image')
        axes[i, 0].axis('off')
        
        # Ground truth mask
        axes[i, 1].imshow(masks[i, 0], cmap='gray')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Prediction
        axes[i, 2].imshow(predictions[i, 0], cmap='gray')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_sample(image, mask, prediction=None):
    """
    Visualize a single sample with image, mask, and optional prediction
    
    Args:
        image (torch.Tensor or np.ndarray): Input image
        mask (torch.Tensor or np.ndarray): Ground truth mask
        prediction (torch.Tensor or np.ndarray, optional): Predicted mask
    """
    # Convert to numpy if needed
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()
    if prediction is not None and torch.is_tensor(prediction):
        prediction = prediction.detach().cpu().numpy()
    
    # Remove channel dimension if present
    if image.ndim == 3:
        image = image[0]
    if mask.ndim == 3:
        mask = mask[0]
    if prediction is not None and prediction.ndim == 3:
        prediction = prediction[0]
    
    # Create figure
    num_cols = 3 if prediction is not None else 2
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 5))
    
    # Image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Ground Truth')
    axes[1].axis('off')
    
    # Prediction
    if prediction is not None:
        axes[2].imshow(prediction, cmap='gray')
        axes[2].set_title('Prediction')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test visualization
    test_image = torch.randn(4, 1, 256, 256)
    test_mask = torch.randint(0, 2, (4, 1, 256, 256)).float()
    test_pred = torch.rand(4, 1, 256, 256)
    
    save_prediction_grid(test_image, test_mask, test_pred, 'test_visualization.png', num_samples=2)
    print("Test visualization saved!")
