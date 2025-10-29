"""
Sorensen-Dice Loss Function for Binary Segmentation
"""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """
    Sorensen-Dice Loss for binary segmentation.
    
    Formula: 1 - (2 * intersection + smooth) / (sum(y_true) + sum(y_pred) + smooth)
    """
    
    def __init__(self, smooth: float = 1e-6):
        """
        Args:
            smooth (float): Smoothing factor to prevent division by zero. Default: 1e-6
        """
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            y_pred (torch.Tensor): Predicted segmentation mask (B, C, H, W) or (B, H, W)
            y_true (torch.Tensor): Ground truth mask (B, C, H, W) or (B, H, W)
        
        Returns:
            torch.Tensor: Dice loss value
        """
        # Flatten tensors
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        # Calculate intersection
        intersection = torch.sum(y_true_flat * y_pred_flat)
        
        # Calculate Dice coefficient
        dice_coefficient = (2.0 * intersection + self.smooth) / (
            torch.sum(y_true_flat) + torch.sum(y_pred_flat) + self.smooth
        )
        
        # Dice loss is 1 - Dice coefficient
        dice_loss = 1.0 - dice_coefficient
        
        return dice_loss


def dice_loss(y_pred: torch.Tensor, y_true: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Functional API for Dice loss.
    
    Args:
        y_pred (torch.Tensor): Predicted segmentation mask
        y_true (torch.Tensor): Ground truth mask
        smooth (float): Smoothing factor to prevent division by zero. Default: 1e-6
    
    Returns:
        torch.Tensor: Dice loss value
    """
    # Flatten tensors
    y_pred_flat = y_pred.view(-1)
    y_true_flat = y_true.view(-1)
    
    # Calculate intersection
    intersection = torch.sum(y_true_flat * y_pred_flat)
    
    # Calculate Dice coefficient
    dice_coefficient = (2.0 * intersection + smooth) / (
        torch.sum(y_true_flat) + torch.sum(y_pred_flat) + smooth
    )
    
    # Dice loss is 1 - Dice coefficient
    return 1.0 - dice_coefficient
