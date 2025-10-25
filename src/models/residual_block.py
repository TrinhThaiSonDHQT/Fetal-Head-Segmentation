"""
Residual Block for Improved U-Net Architecture

This module implements the residual block used in both the encoder and decoder
of the improved U-Net model for fetal head segmentation.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    Residual Block with two convolutional layers and a skip connection.
    
    Architecture:
        - Conv2d(in_channels, out_channels, 3x3) + BatchNorm + Sigmoid
        - Conv2d(out_channels, out_channels, 3x3) + BatchNorm + Tanh
        - Skip connection (with 1x1 conv if channels differ)
        - Final activation after addition
    
    Uses sigmoid-tanh activation functions for improved feature learning.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        activation (str): Activation function pair to use ('sigmoid-tanh' or 'relu')
                         Default: 'sigmoid-tanh'
    """
    
    def __init__(self, in_channels: int, out_channels: int, activation: str = 'sigmoid-tanh'):
        super().__init__()
        
        # Select activation function based on configuration
        self.activation_type = activation.lower()
        
        if self.activation_type == 'sigmoid-tanh':
            # Sigmoid for first conv, tanh for second conv
            self.activation1 = nn.Sigmoid()
            self.activation2 = nn.Tanh()
            self.final_activation = nn.Tanh()
        elif self.activation_type == 'relu':
            # Alternative: ReLU for all (previous implementation)
            self.activation1 = nn.ReLU(inplace=True)
            self.activation2 = nn.ReLU(inplace=True)
            self.final_activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'sigmoid-tanh' or 'relu'.")
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity mapping)
        # Use 1x1 convolution if input and output channels differ
        self.skip_connection = nn.Sequential()
        if in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Final activation after residual addition
        if self.activation_type == 'relu':
            self.final_activation = nn.ReLU(inplace=True)
        elif self.activation_type == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        else:  # tanh
            self.final_activation = nn.Tanh()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width)
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width)
        """
        # Save identity for skip connection
        identity = self.skip_connection(x)
        
        # First convolutional block with sigmoid activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation1(out)
        
        # Second convolutional block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add skip connection (residual) BEFORE final activation
        out = out + identity
        
        # Single final activation after residual addition
        # This maintains gradient flow and prevents double squashing
        out = self.activation2(out)
        
        return out


if __name__ == "__main__":
    # Test the ResidualBlock
    print("Testing ResidualBlock...")
    
    # Test with matching channels (sigmoid-tanh activation - default)
    block1 = ResidualBlock(64, 64, activation='sigmoid-tanh')
    x1 = torch.randn(2, 64, 128, 128)
    y1 = block1(x1)
    print(f"Input shape: {x1.shape} -> Output shape: {y1.shape}")
    assert y1.shape == x1.shape, "Output shape mismatch for same channels"
    
    # Test with different channels
    block2 = ResidualBlock(64, 128, activation='sigmoid-tanh')
    x2 = torch.randn(2, 64, 128, 128)
    y2 = block2(x2)
    print(f"Input shape: {x2.shape} -> Output shape: {y2.shape}")
    assert y2.shape == (2, 128, 128, 128), "Output shape mismatch for different channels"
    
    # Test with ReLU activation (alternative)
    block3 = ResidualBlock(32, 64, activation='relu')
    x3 = torch.randn(2, 32, 64, 64)
    y3 = block3(x3)
    print(f"Input shape (relu): {x3.shape} -> Output shape: {y3.shape}")
    
    print("\n✓ All tests passed!")

    # Count parameters
    total_params = sum(p.numel() for p in block1.parameters())
    print(f"\nTotal parameters in ResidualBlock(64->64): {total_params:,}")
    
    print("\n✓ All tests passed!")
