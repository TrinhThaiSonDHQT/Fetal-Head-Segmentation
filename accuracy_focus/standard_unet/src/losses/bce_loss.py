"""
Binary Cross-Entropy Loss for Standard U-Net

Standard loss function as used in the original U-Net paper.
"""

import torch
import torch.nn as nn


class BCELoss(nn.Module):
    """
    Simple Binary Cross-Entropy Loss wrapper.
    
    For standard U-Net baseline. Model outputs probabilities (after sigmoid).
    
    Args:
        None
    """
    
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bce = nn.BCELoss()
    
    def forward(self, pred, target):
        """
        Calculate BCE loss.
        
        Args:
            pred (torch.Tensor): Model predictions AFTER sigmoid (probabilities).
                                 Shape: (B, 1, H, W), Range: [0, 1]
            target (torch.Tensor): Ground truth binary masks.
                                   Shape: (B, 1, H, W), Range: [0, 1]
        
        Returns:
            torch.Tensor: BCE loss value (scalar)
        """
        return self.bce(pred, target)


if __name__ == "__main__":
    # Test the loss
    criterion = BCELoss()
    
    # Create dummy data
    pred = torch.sigmoid(torch.randn(2, 1, 256, 256))  # Probabilities
    target = torch.randint(0, 2, (2, 1, 256, 256)).float()
    
    loss = criterion(pred, target)
    print(f"BCE Loss: {loss.item():.4f}")
