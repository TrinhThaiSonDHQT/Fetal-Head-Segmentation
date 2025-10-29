"""
Convolutional Block for Attention U-Net
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """
    Convolutional block consisting of two sequential operations:
    (3x3 Conv2D + Batch Normalization + ReLU activation)
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels (filters)
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super(ConvBlock, self).__init__()
        
        # First convolution block
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=False  # No bias needed before BatchNorm
        )
        self.bn1 = nn.BatchNorm2d(num_features=out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # Second convolution block
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,  # 'same' padding
            bias=False  # No bias needed before BatchNorm
        )
        self.bn2 = nn.BatchNorm2d(num_features=out_channels)
        self.relu2 = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ConvBlock.
        
        Args:
            x (torch.Tensor): Input tensor (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output tensor (B, C_out, H, W)
        """
        # First: Conv -> BN -> ReLU
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        # Second: Conv -> BN -> ReLU
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


def conv_block(x: torch.Tensor, filters: int, in_channels: int = None) -> torch.Tensor:
    """
    Helper function to create and apply a convolutional block.
    
    Args:
        x (torch.Tensor): Input tensor (B, C_in, H, W)
        filters (int): Number of output filters
        in_channels (int, optional): Number of input channels. If None, inferred from x
    
    Returns:
        torch.Tensor: Output tensor (B, filters, H, W)
    """
    if in_channels is None:
        in_channels = x.shape[1]
    
    block = ConvBlock(in_channels=in_channels, out_channels=filters)
    
    # Move block to same device as input
    block = block.to(x.device)
    
    return block(x)


if __name__ == "__main__":
    # Test the ConvBlock module
    batch_size = 2
    in_channels = 64
    out_channels = 128
    height, width = 256, 256
    
    # Create conv block
    conv_blk = ConvBlock(in_channels=in_channels, out_channels=out_channels)
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, height, width)
    
    # Forward pass
    output = conv_blk(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Spatial dimensions preserved: {output.shape[2:] == x.shape[2:]}")
    print(f"Output channels correct: {output.shape[1] == out_channels}")
    
    # Count parameters
    total_params = sum(p.numel() for p in conv_blk.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test helper function
    print("\n--- Testing helper function ---")
    x_test = torch.randn(1, 32, 128, 128)
    output_test = conv_block(x_test, filters=64)
    print(f"Helper function input shape: {x_test.shape}")
    print(f"Helper function output shape: {output_test.shape}")
