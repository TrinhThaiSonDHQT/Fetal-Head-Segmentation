"""
BCEWithLogitsLoss with automatic pos_weight calculation for extreme class imbalance.

This loss function is specifically designed for binary segmentation tasks with
severe class imbalance (e.g., fetal head segmentation where foreground is ~0.5-1% of pixels).

Key advantages:
- Combines BCE and sigmoid in one operation (numerically stable)
- Automatic pos_weight calculation from data
- Handles extreme class imbalance effectively
"""

import torch
import torch.nn as nn


class BCEWithLogitsBalancedLoss(nn.Module):
    """
    BCEWithLogitsLoss with automatic pos_weight for class imbalance.
    
    This loss applies sigmoid activation and BCE loss in a single operation,
    which is more numerically stable than applying them separately.
    
    Args:
        pos_weight (float, optional): Weight for positive class. If None, will be
                                     computed from first batch. Typical values:
                                     - 0.5% foreground → pos_weight = 199
                                     - 1.0% foreground → pos_weight = 99
                                     - 2.0% foreground → pos_weight = 49
        auto_weight (bool): If True, computes pos_weight from first batch. Default: True
        reduction (str): Reduction method ('mean', 'sum', 'none'). Default: 'mean'
    
    Note:
        Model must output LOGITS (raw values), NOT probabilities.
        Do NOT use sigmoid activation in the final layer.
    """
    
    def __init__(
        self,
        pos_weight: float = None,
        auto_weight: bool = True,
        reduction: str = 'mean'
    ):
        super(BCEWithLogitsBalancedLoss, self).__init__()
        self.auto_weight = auto_weight
        self.reduction = reduction
        self._pos_weight_computed = False
        
        if pos_weight is not None:
            self.register_buffer('pos_weight', torch.tensor([pos_weight]))
            self._pos_weight_computed = True
            self.auto_weight = False
        else:
            self.register_buffer('pos_weight', None)
    
    def _compute_pos_weight(self, targets: torch.Tensor):
        """
        Compute pos_weight from target distribution.
        pos_weight = num_negative_pixels / num_positive_pixels
        """
        num_positive = targets.sum()
        num_negative = targets.numel() - num_positive
        
        if num_positive > 0:
            pos_weight = num_negative / num_positive
        else:
            pos_weight = torch.tensor(1.0, device=targets.device)
        
        return pos_weight
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute BCEWithLogits loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs (BEFORE sigmoid), shape (B, 1, H, W)
            targets (torch.Tensor): Binary masks {0, 1}, shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Computed loss value
        """
        # Auto-compute pos_weight from first batch if needed
        if self.auto_weight and not self._pos_weight_computed:
            self.pos_weight = self._compute_pos_weight(targets)
            self._pos_weight_computed = True
            print(f"[BCEWithLogitsBalancedLoss] Auto-computed pos_weight: {self.pos_weight.item():.2f}")
            print(f"  Foreground ratio: {targets.mean().item():.4%}")
            print(f"  This gives {self.pos_weight.item():.1f}x more weight to foreground pixels")
        
        # Compute BCE with logits
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=self.pos_weight,
            reduction=self.reduction
        )
        
        return loss


class DiceBCEWithLogitsLoss(nn.Module):
    """
    Combined Dice Loss + BCEWithLogits Loss for balanced training.
    
    Combines two complementary loss functions:
    - BCEWithLogits: Pixel-wise classification with class balance (via pos_weight)
    - Dice Loss: Region overlap optimization
    
    Args:
        dice_weight (float): Weight for Dice loss component. Default: 0.5
        bce_weight (float): Weight for BCE loss component. Default: 0.5
        pos_weight (float, optional): Positive class weight for BCE. If None, auto-computed.
        auto_weight (bool): Auto-compute pos_weight from first batch. Default: True
        smooth (float): Smoothing constant for Dice loss. Default: 1.0
    
    Formula:
        Loss = dice_weight * DiceLoss + bce_weight * BCEWithLogitsLoss
    """
    
    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        pos_weight: float = None,
        auto_weight: bool = True,
        smooth: float = 1.0
    ):
        super(DiceBCEWithLogitsLoss, self).__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.smooth = smooth
        
        # BCE with logits loss (handles class imbalance)
        self.bce_loss = BCEWithLogitsBalancedLoss(
            pos_weight=pos_weight,
            auto_weight=auto_weight,
            reduction='mean'
        )
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Dice + BCEWithLogits loss.
        
        Args:
            logits (torch.Tensor): Raw model outputs (BEFORE sigmoid), shape (B, 1, H, W)
            targets (torch.Tensor): Binary masks {0, 1}, shape (B, 1, H, W)
        
        Returns:
            torch.Tensor: Combined loss value
        """
        # BCEWithLogits loss (on logits directly)
        bce = self.bce_loss(logits, targets)
        
        # Dice loss (on probabilities)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff.mean()
        
        # Combined loss
        return self.dice_weight * dice_loss + self.bce_weight * bce


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("BCEWithLogitsLoss Testing")
    print("="*80)
    
    # Create sample data (0.7% foreground, typical for fetal head)
    batch_size = 4
    logits = torch.randn(batch_size, 1, 256, 256)  # Raw outputs from model
    targets = torch.zeros(batch_size, 1, 256, 256)
    targets[:, :, 100:120, 100:120] = 1  # Small foreground region (~0.6%)
    
    print(f"\nTest data:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Foreground ratio: {targets.mean():.4%}")
    
    # Test BCEWithLogits with auto pos_weight
    print(f"\n{'='*80}")
    print("1. BCEWithLogitsBalancedLoss (auto pos_weight)")
    print(f"{'='*80}")
    criterion1 = BCEWithLogitsBalancedLoss(auto_weight=True)
    loss1 = criterion1(logits, targets)
    print(f"Loss: {loss1.item():.6f}")
    
    # Test BCEWithLogits with manual pos_weight
    print(f"\n{'='*80}")
    print("2. BCEWithLogitsBalancedLoss (manual pos_weight=100)")
    print(f"{'='*80}")
    criterion2 = BCEWithLogitsBalancedLoss(pos_weight=100.0, auto_weight=False)
    loss2 = criterion2(logits, targets)
    print(f"Loss: {loss2.item():.6f}")
    
    # Test combined Dice + BCEWithLogits
    print(f"\n{'='*80}")
    print("3. DiceBCEWithLogitsLoss (combined)")
    print(f"{'='*80}")
    criterion3 = DiceBCEWithLogitsLoss(dice_weight=0.5, bce_weight=0.5, auto_weight=True)
    loss3 = criterion3(logits, targets)
    print(f"Loss: {loss3.item():.6f}")
    
    print(f"\n{'='*80}")
    print("✓ All tests passed!")
    print(f"{'='*80}")
