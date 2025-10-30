"""
Focal Tversky Loss for extreme class imbalance in medical image segmentation.

This loss function is specifically designed for tasks where the foreground class
is extremely small (e.g., 0.5% of pixels). It combines:
- Tversky Index: Asymmetric similarity measure with controllable FN/FP trade-off
- Focal mechanism: Down-weights easy examples to focus on hard cases

Reference:
    Abraham, N., & Khan, N. M. (2019). 
    "A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation"
    IEEE International Symposium on Biomedical Imaging (ISBI).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for binary segmentation with extreme class imbalance.
    
    Formula:
        TI = TP / (TP + alpha*FN + beta*FP)
        FTL = (1 - TI)^gamma
    
    Args:
        alpha (float): Weight for false negatives. Higher alpha penalizes missing foreground.
                      For 0.5% foreground, use alpha=0.7 (prioritize recall).
                      Range: [0, 1], typically 0.5-0.7 for extreme imbalance.
        beta (float): Weight for false positives. Lower beta is more tolerant of false alarms.
                     For extreme imbalance, use beta=0.3 (1-alpha).
                     Range: [0, 1], typically 0.3-0.5.
        gamma (float): Focal parameter to down-weight easy examples.
                      Higher gamma focuses more on hard cases.
                      Range: [1, 3], typically 1.33-2.0.
                      - gamma=1.0: equivalent to Tversky Loss
                      - gamma=1.33: balanced focusing (recommended start)
                      - gamma>2.0: very aggressive, may cause instability
        smooth (float): Smoothing constant to avoid division by zero.
                       Range: [1e-6, 1.0], typically 1.0 for stability.
    
    Shape:
        - Input: (N, 1, H, W) or (N, H, W) - predicted probabilities [0, 1]
        - Target: (N, 1, H, W) or (N, H, W) - binary ground truth {0, 1}
        - Output: scalar loss value
    
    Example:
        >>> criterion = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        >>> pred = torch.sigmoid(model(input))  # Apply sigmoid to logits
        >>> loss = criterion(pred, target)
    """
    
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1.0):
        super(FocalTverskyLoss, self).__init__()
        
        # Validate parameters
        assert 0 <= alpha <= 1, f"alpha must be in [0, 1], got {alpha}"
        assert 0 <= beta <= 1, f"beta must be in [0, 1], got {beta}"
        assert gamma >= 1, f"gamma must be >= 1, got {gamma}"
        assert smooth > 0, f"smooth must be > 0, got {smooth}"
        
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Focal Tversky Loss.
        
        Args:
            pred: Predicted probabilities after sigmoid, shape (N, 1, H, W) or (N, H, W)
            target: Ground truth binary mask, shape (N, 1, H, W) or (N, H, W)
        
        Returns:
            Scalar loss value
        """
        # Ensure inputs have same shape
        if pred.dim() == 4 and pred.size(1) == 1:
            pred = pred.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
        if target.dim() == 4 and target.size(1) == 1:
            target = target.squeeze(1)  # (N, 1, H, W) -> (N, H, W)
        
        # Flatten spatial dimensions: (N, H, W) -> (N, H*W)
        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)
        
        # Ensure target is float
        target = target.float()
        
        # Compute true positives, false negatives, false positives
        # TP: predicted=1, actual=1
        # FN: predicted=0, actual=1
        # FP: predicted=1, actual=0
        tp = (pred * target).sum(dim=1)  # (N,)
        fn = ((1 - pred) * target).sum(dim=1)  # (N,)
        fp = (pred * (1 - target)).sum(dim=1)  # (N,)
        
        # Compute Tversky Index per sample
        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        
        # Apply focal mechanism
        focal_tversky = torch.pow(1 - tversky_index, self.gamma)
        
        # Average over batch
        loss = focal_tversky.mean()
        
        return loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"alpha={self.alpha}, beta={self.beta}, "
                f"gamma={self.gamma}, smooth={self.smooth})")


class CombinedFocalTverskyLoss(nn.Module):
    """
    Combined loss: Focal Tversky + Dice + BCE (Option 2 from papers).
    
    Use this if pure Focal Tversky plateaus at 85-90% Dice after proper training.
    
    Formula:
        Loss = w_ftl * FTL + w_dice * Dice + w_bce * BCE
    
    Args:
        ftl_weight (float): Weight for Focal Tversky Loss (default: 0.5)
        dice_weight (float): Weight for Dice Loss (default: 0.3)
        bce_weight (float): Weight for BCE Loss (default: 0.2)
        alpha (float): FTL alpha parameter (default: 0.7)
        beta (float): FTL beta parameter (default: 0.3)
        gamma (float): FTL gamma parameter (default: 1.5)
        pos_weight (float): BCE positive class weight (default: 300 for 0.5% foreground)
    
    Example:
        >>> criterion = CombinedFocalTverskyLoss(
        ...     ftl_weight=0.5, dice_weight=0.3, bce_weight=0.2,
        ...     alpha=0.7, gamma=1.5, pos_weight=300
        ... )
        >>> # Apply sigmoid BEFORE passing to loss
        >>> pred_probs = torch.sigmoid(logits)
        >>> loss = criterion(pred_probs, logits, target)
    """
    
    def __init__(
        self,
        ftl_weight: float = 0.5,
        dice_weight: float = 0.3,
        bce_weight: float = 0.2,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 1.5,
        pos_weight: float = 300.0,
        smooth: float = 1.0
    ):
        super(CombinedFocalTverskyLoss, self).__init__()
        
        # Validate weights sum to 1.0
        total_weight = ftl_weight + dice_weight + bce_weight
        assert abs(total_weight - 1.0) < 1e-5, f"Weights must sum to 1.0, got {total_weight}"
        
        self.ftl_weight = ftl_weight
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        
        # Loss components
        self.focal_tversky = FocalTverskyLoss(alpha=alpha, beta=beta, gamma=gamma, smooth=smooth)
        self.smooth = smooth
        
        # BCE with positive weight for class imbalance
        self.bce_pos_weight = torch.tensor([pos_weight])
        
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice Loss."""
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target.view(target.size(0), -1).float()
        
        intersection = (pred_flat * target_flat).sum(dim=1)
        union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return (1 - dice).mean()
    
    def forward(
        self,
        pred_probs: torch.Tensor,
        logits: torch.Tensor,
        target: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred_probs: Predicted probabilities after sigmoid, shape (N, 1, H, W)
            logits: Raw model output before sigmoid, shape (N, 1, H, W)
            target: Ground truth binary mask, shape (N, 1, H, W)
        
        Returns:
            Combined scalar loss
        """
        # Move pos_weight to same device as logits
        if self.bce_pos_weight.device != logits.device:
            self.bce_pos_weight = self.bce_pos_weight.to(logits.device)
        
        # Compute individual losses
        ftl = self.focal_tversky(pred_probs, target)
        dice = self.dice_loss(pred_probs, target)
        bce = F.binary_cross_entropy_with_logits(
            logits, target.float(),
            pos_weight=self.bce_pos_weight
        )
        
        # Weighted combination
        total_loss = (
            self.ftl_weight * ftl +
            self.dice_weight * dice +
            self.bce_weight * bce
        )
        
        return total_loss
    
    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"ftl_weight={self.ftl_weight}, dice_weight={self.dice_weight}, "
                f"bce_weight={self.bce_weight})")


# Convenience function for quick setup
def get_loss_function(loss_type: str = "focal_tversky", **kwargs):
    """
    Factory function to get loss function by name.
    
    Args:
        loss_type: One of ["focal_tversky", "combined"]
        **kwargs: Parameters passed to loss constructor
    
    Returns:
        Loss function instance
    
    Example:
        >>> # Option 1: Pure Focal Tversky (recommended start)
        >>> criterion = get_loss_function("focal_tversky", alpha=0.7, gamma=1.33)
        >>> 
        >>> # Option 2: Combined loss (if Option 1 plateaus)
        >>> criterion = get_loss_function("combined", ftl_weight=0.5, pos_weight=300)
    """
    if loss_type == "focal_tversky":
        return FocalTverskyLoss(**kwargs)
    elif loss_type == "combined":
        return CombinedFocalTverskyLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}. Choose 'focal_tversky' or 'combined'")
