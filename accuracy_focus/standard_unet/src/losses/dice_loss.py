"""
Dice Loss Implementation for Binary Segmentation

This module provides a Dice Loss function specifically designed for handling
class imbalance in binary segmentation tasks (e.g., fetal head segmentation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    
    The Dice coefficient measures the overlap between predicted and ground truth masks.
    Dice Loss = 1 - Dice Coefficient, encouraging the model to maximize overlap.
    
    Args:
        smooth (float): Smoothing constant to avoid division by zero. Default: 1.0
        reduction (str): Specifies the reduction to apply to the output.
                        Options: 'mean', 'sum', 'none'. Default: 'mean'
    
    Formula:
        Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
        Loss = 1 - Dice
    """
    
    def __init__(self, smooth: float = 1.0, reduction: str = 'mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice Loss.
        
        Args:
            predictions (torch.Tensor): Model predictions with shape (B, 1, H, W)
                                       Values should be in range [0, 1] (after sigmoid)
            targets (torch.Tensor): Ground truth masks with shape (B, 1, H, W)
                                   Binary values {0, 1}
        
        Returns:
            torch.Tensor: Computed Dice loss
        """
        # Flatten spatial dimensions: (B, 1, H, W) -> (B, H*W)
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        
        # Compute Dice coefficient per sample
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Compute Dice loss
        dice_loss = 1.0 - dice_coeff
        
        # Apply reduction
        if self.reduction == 'mean':
            return dice_loss.mean()
        elif self.reduction == 'sum':
            return dice_loss.sum()
        elif self.reduction == 'none':
            return dice_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss + Binary Cross-Entropy Loss.
    
    This hybrid loss combines the benefits of both:
    - Dice Loss: Handles class imbalance and focuses on overlap
    - BCE Loss: Provides pixel-wise supervision and stable gradients
    
    Args:
        dice_weight (float): Weight for Dice loss component. Default: 0.5
        bce_weight (float): Weight for BCE loss component. Default: 0.5
        smooth (float): Smoothing constant for Dice loss. Default: 1.0
    
    Formula:
        Loss = dice_weight * DiceLoss + bce_weight * BCELoss
    """
    
    def __init__(self, dice_weight: float = 0.5, bce_weight: float = 0.5, smooth: float = 1.0):
        super(DiceBCELoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice_loss = DiceLoss(smooth=smooth)
        self.bce_loss = nn.BCELoss()
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Dice + BCE loss.
        
        Args:
            predictions (torch.Tensor): Model predictions with shape (B, 1, H, W)
            targets (torch.Tensor): Ground truth masks with shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Combined loss value
        """
        dice = self.dice_loss(predictions, targets)
        bce = self.bce_loss(predictions, targets)
        
        return self.dice_weight * dice + self.bce_weight * bce


class FocalDiceLoss(nn.Module):
    """
    Focal Dice Loss - Dice Loss with focal weighting.
    
    Applies focal weighting to emphasize hard-to-segment regions,
    particularly useful for small objects like fetal heads.
    
    Args:
        alpha (float): Focal weight exponent. Higher values focus more on hard examples.
                      Default: 0.5
        smooth (float): Smoothing constant. Default: 1.0
    """
    
    def __init__(self, alpha: float = 0.5, smooth: float = 1.0):
        super(FocalDiceLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Dice Loss.
        
        Args:
            predictions (torch.Tensor): Model predictions with shape (B, 1, H, W)
            targets (torch.Tensor): Ground truth masks with shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Computed focal Dice loss
        """
        # Flatten spatial dimensions
        predictions = predictions.view(predictions.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Compute intersection and union
        intersection = (predictions * targets).sum(dim=1)
        union = predictions.sum(dim=1) + targets.sum(dim=1)
        
        # Compute Dice coefficient
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Apply focal weighting: emphasize samples with low Dice scores
        focal_weight = (1.0 - dice_coeff) ** self.alpha
        dice_loss = (1.0 - dice_coeff) * focal_weight
        
        return dice_loss.mean()


# Factory function for easy loss selection
def get_loss_function(loss_type: str = 'dice', **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type (str): Type of loss. Options: 'dice', 'dice_bce', 'focal_dice'
        **kwargs: Additional arguments passed to the loss constructor
    
    Returns:
        nn.Module: Instantiated loss function
    
    Example:
        >>> loss_fn = get_loss_function('dice', smooth=1.0)
        >>> loss_fn = get_loss_function('dice_bce', dice_weight=0.6, bce_weight=0.4)
    """
    loss_dict = {
        'dice': DiceLoss,
        'dice_bce': DiceBCELoss,
        'focal_dice': FocalDiceLoss
    }
    
    if loss_type not in loss_dict:
        raise ValueError(f"Unknown loss type: {loss_type}. Available: {list(loss_dict.keys())}")
    
    return loss_dict[loss_type](**kwargs)


if __name__ == "__main__":
    # Quick test
    print("Testing Dice Loss implementations...\n")
    
    # Create dummy data
    batch_size, channels, height, width = 4, 1, 256, 256
    predictions = torch.rand(batch_size, channels, height, width)  # After sigmoid
    targets = torch.randint(0, 2, (batch_size, channels, height, width)).float()
    
    # Test DiceLoss
    dice_loss_fn = DiceLoss(smooth=1.0)
    dice_loss_value = dice_loss_fn(predictions, targets)
    print(f"DiceLoss: {dice_loss_value.item():.4f}")
    
    # Test DiceBCELoss
    dice_bce_loss_fn = DiceBCELoss(dice_weight=0.5, bce_weight=0.5)
    dice_bce_loss_value = dice_bce_loss_fn(predictions, targets)
    print(f"DiceBCELoss: {dice_bce_loss_value.item():.4f}")
    
    # Test FocalDiceLoss
    focal_dice_loss_fn = FocalDiceLoss(alpha=0.5)
    focal_dice_loss_value = focal_dice_loss_fn(predictions, targets)
    print(f"FocalDiceLoss: {focal_dice_loss_value.item():.4f}")
    
    print("\n✓ All loss functions working correctly!")
