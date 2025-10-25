"""
Scale Attention Module (SAM) for Improved U-Net
"""
import torch
import torch.nn as nn


class ScaleAttentionModule(nn.Module):
    """
    Scale Attention Module that adaptively weights different feature scales.
    
    Uses squeeze-and-excitation mechanism to generate attention weights that
    reduce noise and select the most useful features.
    
    Args:
        in_channels (int): Number of input channels
        reduction_ratio (int): Ratio for channel reduction in attention mechanism (default=16)
    """
    
    def __init__(self, in_channels, reduction_ratio=16):
        super(ScaleAttentionModule, self).__init__()
        
        # Ensure reduction doesn't make channels less than 1
        reduced_channels = max(in_channels // reduction_ratio, 1)
        
        # Global context extraction with squeeze-and-excitation
        self.attention = nn.Sequential(
            # Squeeze: Global average pooling
            nn.AdaptiveAvgPool2d(1),
            
            # Excitation: Channel reduction
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            
            # Excitation: Channel expansion
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            
            # Generate attention weights
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through Scale Attention Module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Attention-weighted output of shape (B, C, H, W)
        """
        # Generate attention weights
        attention_weights = self.attention(x)
        
        # Apply attention weights to input
        out = x * attention_weights
        
        return out
