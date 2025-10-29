"""
Hybrid Loss Function: Dice Loss + Binary Cross-Entropy Loss

This module implements a combined loss function for binary segmentation tasks.
The hybrid loss combines BCE (pixel-wise accuracy) and Dice (region overlap).

Formula:
    Loss = w_BCE × Loss_BCE + w_Dice × Loss_Dice
    where Dice Loss = 1 - DSC
    and DSC = (2 * |X ∩ Y|) / (|X| + |Y|)
"""

import torch
import torch.nn as nn


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross-Entropy Loss.
    
    This loss function combines:
    - BCE Loss: For pixel-wise accuracy (with optional weighting for class imbalance)
    - Dice Loss: For region overlap optimization
    
    Args:
        smooth (float): Smoothing constant to avoid division by zero. Default: 1e-6
        dice_weight (float): Weight for Dice loss component. Default: 0.5
        bce_weight (float): Weight for BCE loss component. Default: 0.5
    """
    
    def __init__(self, smooth=1e-6, dice_weight=0.8, bce_weight=0.2, pos_weight=200.0):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        # Handle pos_weight (can be float or tensor)
        if isinstance(pos_weight, torch.Tensor):
            # Already a tensor, use as-is (device will be preserved)
            pos_weight_tensor = pos_weight
        else:
            # Convert float to tensor (will be moved to correct device in forward pass)
            pos_weight_tensor = torch.tensor([pos_weight])
        
        # Weighted BCE: heavily penalize false negatives (missing fetal head)
        # pos_weight=200 because background:foreground ratio is ~260:1
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    def forward(self, pred, target):
        """
        Forward pass of the loss function.
        
        Args:
            pred (torch.Tensor): Model predictions BEFORE sigmoid (logits).
                                 Shape: (B, 1, H, W)
            target (torch.Tensor): Ground truth binary masks.
                                   Shape: (B, 1, H, W)
        
        Returns:
            torch.Tensor: Combined loss value (scalar)
        """
        # Ensure pos_weight is on the same device as input tensors
        if self.bce.pos_weight.device != pred.device:
            self.bce.pos_weight = self.bce.pos_weight.to(pred.device)
        
        # Calculate BCE loss (BCEWithLogitsLoss expects raw logits)
        bce_loss = self.bce(pred, target)
        
        # Apply sigmoid for Dice calculation
        pred_sigmoid = torch.sigmoid(pred)
        
        # Calculate Dice loss
        pred_flat = pred_sigmoid.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_coeff = (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        dice_loss = 1 - dice_coeff
        
        # Combined loss with configurable weights
        total_loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        
        return total_loss
