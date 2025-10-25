"""
ASPP (Atrous Spatial Pyramid Pooling) Module for Improved U-Net

Uses sigmoid and tanh activation functions for enhanced feature extraction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling module for multi-scale feature extraction.
    
    Uses 5 parallel branches:
    - 1x1 convolution
    - 3x3 atrous convolutions with rates 6, 12, 18
    - Global average pooling with 1x1 convolution
    
    Employs Sigmoid-Tanh activation functions for improved gradient flow.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels for each branch
        activation (str): Activation function to use ('sigmoid-tanh' or 'relu')
                         Default: 'sigmoid-tanh'
    """
    
    def __init__(self, in_channels, out_channels, activation='sigmoid-tanh'):
        super(ASPP, self).__init__()
        
        self.activation_type = activation.lower()
        
        # Select activation functions
        if self.activation_type == 'sigmoid-tanh':
            act1 = nn.Sigmoid()
            act2 = nn.Tanh()
        else:  # relu
            act1 = nn.ReLU(inplace=True)
            act2 = nn.ReLU(inplace=True)
        
        # Branch 1: 1x1 convolution
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act1
        )
        
        # Branch 2: 3x3 atrous convolution (rate=6)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_channels),
            act2
        )
        
        # Branch 3: 3x3 atrous convolution (rate=12)
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_channels),
            act1
        )
        
        # Branch 4: 3x3 atrous convolution (rate=18)
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18, bias=False),
            nn.BatchNorm2d(out_channels),
            act2
        )
        
        # Branch 5: Image pooling (global average pooling + 1x1 conv)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act1
        )
        
        # Final 1x1 convolution to fuse all branches
        self.conv_concat = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act2
        )
    
    def forward(self, x):
        """
        Forward pass through ASPP module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        
        Returns:
            torch.Tensor: Output tensor of shape (B, out_channels, H, W)
        """
        # Get spatial dimensions
        size = x.shape[2:]
        
        # Process through all branches
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        # Image pooling branch needs upsampling
        out5 = self.branch5(x)
        out5 = F.interpolate(out5, size=size, mode='bilinear', align_corners=False)
        
        # Concatenate all branches
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        
        # Final fusion convolution
        out = self.conv_concat(out)
        
        return out
