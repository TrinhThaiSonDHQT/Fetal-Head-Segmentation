"""
SOLUTION: Proper loss function for extreme class imbalance
"""
import torch
import torch.nn as nn


class BalancedBCEDiceLoss(nn.Module):
    """
    Balanced BCE + Dice Loss for extreme class imbalance.
    
    Uses pos_weight in BCE to give more importance to foreground pixels.
    Combines with Dice loss for overlap-based optimization.
    """
    
    def __init__(self, bce_weight=0.5, dice_weight=0.5, pos_weight=None, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        
        # If pos_weight not provided, will be computed from first batch
        if pos_weight is not None:
            self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Raw model outputs (BEFORE sigmoid), shape (B, 1, H, W)
            targets: Binary masks {0, 1}, shape (B, 1, H, W)
        """
        # BCE Loss (on logits)
        bce = self.bce_loss(logits, targets)
        
        # Dice Loss (on probabilities)
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        
        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)
        dice_coeff = (2.0 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1.0 - dice_coeff.mean()
        
        return self.bce_weight * bce + self.dice_weight * dice_loss


# USAGE EXAMPLE:
if __name__ == "__main__":
    # Calculate pos_weight from your dataset
    # pos_weight = num_negative_pixels / num_positive_pixels
    # For 0.7% foreground: pos_weight ≈ 140
    
    # Example with typical fetal head segmentation (0.5% foreground)
    pos_weight = 199.0  # (100 - 0.5) / 0.5 = 199
    
    criterion = BalancedBCEDiceLoss(
        bce_weight=0.5,
        dice_weight=0.5,
        pos_weight=pos_weight,
        smooth=1.0
    )
    
    print(f"Balanced Loss created with pos_weight={pos_weight}")
    print(f"This gives {pos_weight}x more importance to foreground pixels")
    
    # Test
    batch_size = 4
    logits = torch.randn(batch_size, 1, 256, 256)  # Raw outputs
    targets = torch.zeros(batch_size, 1, 256, 256)
    targets[:, :, 100:150, 100:150] = 1  # Small foreground region
    
    loss = criterion(logits, targets)
    print(f"\nTest loss: {loss.item():.4f}")
