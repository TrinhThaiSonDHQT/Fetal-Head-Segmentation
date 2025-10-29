"""
Attention U-Net for BCEWithLogitsLoss (outputs logits, no sigmoid)
This variant is specifically designed to work with BCEWithLogitsLoss for better numerical stability.
"""

import torch
import torch.nn as nn
from typing import Tuple

# Import from the original attention_unet module
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent / 'attention_unet'
sys.path.insert(0, str(parent_dir))

from conv_block import ConvBlock
from attention_gate import AttentionGate


class AttentionUNetLogits(nn.Module):
    """
    2D Attention U-Net that outputs LOGITS (for use with BCEWithLogitsLoss).
    
    This model is identical to AttentionUNet but WITHOUT sigmoid activation.
    Use this with BCEWithLogitsLoss for better numerical stability and handling
    of extreme class imbalance.
    
    Key differences from standard AttentionUNet:
    - NO sigmoid activation in output layer
    - Outputs raw logits instead of probabilities [0, 1]
    - More numerically stable when combined with BCEWithLogitsLoss
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        out_channels (int): Number of output channels (e.g., 1 for binary segmentation)
        base_filters (int): Number of filters in the first encoder block (default: 64)
    """
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_filters: int = 64
    ):
        super(AttentionUNetLogits, self).__init__()
        
        # Define filter sizes for each level
        filters = [base_filters * (2 ** i) for i in range(5)]  # [64, 128, 256, 512, 1024]
        
        # ==================== ENCODER (Down-sampling Path) ====================
        
        # Encoder Block 1
        self.conv_block1 = ConvBlock(in_channels=in_channels, out_channels=filters[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 2
        self.conv_block2 = ConvBlock(in_channels=filters[0], out_channels=filters[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 3
        self.conv_block3 = ConvBlock(in_channels=filters[1], out_channels=filters[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Block 4
        self.conv_block4 = ConvBlock(in_channels=filters[2], out_channels=filters[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = ConvBlock(in_channels=filters[3], out_channels=filters[4])
        
        # ==================== DECODER (Up-sampling Path with Attention) ====================
        
        # Decoder Block 4
        self.up4 = nn.ConvTranspose2d(
            in_channels=filters[4],
            out_channels=filters[3],
            kernel_size=2,
            stride=2
        )
        self.att4 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2], stride=1)
        self.conv_block_up4 = ConvBlock(in_channels=filters[4], out_channels=filters[3])
        
        # Decoder Block 3
        self.up3 = nn.ConvTranspose2d(
            in_channels=filters[3],
            out_channels=filters[2],
            kernel_size=2,
            stride=2
        )
        self.att3 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1], stride=1)
        self.conv_block_up3 = ConvBlock(in_channels=filters[3], out_channels=filters[2])
        
        # Decoder Block 2
        self.up2 = nn.ConvTranspose2d(
            in_channels=filters[2],
            out_channels=filters[1],
            kernel_size=2,
            stride=2
        )
        self.att2 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0], stride=1)
        self.conv_block_up2 = ConvBlock(in_channels=filters[2], out_channels=filters[1])
        
        # Decoder Block 1
        self.up1 = nn.ConvTranspose2d(
            in_channels=filters[1],
            out_channels=filters[0],
            kernel_size=2,
            stride=2
        )
        self.att1 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=filters[0] // 2, stride=1)
        self.conv_block_up1 = ConvBlock(in_channels=filters[1], out_channels=filters[0])
        
        # ==================== OUTPUT LAYER (NO SIGMOID) ====================
        
        self.output_conv = nn.Conv2d(
            in_channels=filters[0],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        # NO SIGMOID - outputs logits for BCEWithLogitsLoss
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention U-Net (Logits version).
        
        Args:
            x (torch.Tensor): Input tensor (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Segmentation LOGITS (B, C_out, H, W) - NOT probabilities
        """
        # ==================== ENCODER ====================
        
        # Block 1
        c1 = self.conv_block1(x)  # (B, 64, H, W)
        p1 = self.pool1(c1)        # (B, 64, H/2, W/2)
        
        # Block 2
        c2 = self.conv_block2(p1)  # (B, 128, H/2, W/2)
        p2 = self.pool2(c2)        # (B, 128, H/4, W/4)
        
        # Block 3
        c3 = self.conv_block3(p2)  # (B, 256, H/4, W/4)
        p3 = self.pool3(c3)        # (B, 256, H/8, W/8)
        
        # Block 4
        c4 = self.conv_block4(p3)  # (B, 512, H/8, W/8)
        p4 = self.pool4(c4)        # (B, 512, H/16, W/16)
        
        # Bottleneck
        bottleneck = self.bottleneck(p4)  # (B, 1024, H/16, W/16)
        
        # ==================== DECODER ====================
        
        # Decoder Block 4
        g4 = self.up4(bottleneck)   # (B, 512, H/8, W/8)
        att4 = self.att4(x=c4, g=g4)  # (B, 512, H/8, W/8)
        concat4 = torch.cat([att4, g4], dim=1)  # (B, 1024, H/8, W/8)
        d4 = self.conv_block_up4(concat4)  # (B, 512, H/8, W/8)
        
        # Decoder Block 3
        g3 = self.up3(d4)           # (B, 256, H/4, W/4)
        att3 = self.att3(x=c3, g=g3)  # (B, 256, H/4, W/4)
        concat3 = torch.cat([att3, g3], dim=1)  # (B, 512, H/4, W/4)
        d3 = self.conv_block_up3(concat3)  # (B, 256, H/4, W/4)
        
        # Decoder Block 2
        g2 = self.up2(d3)           # (B, 128, H/2, W/2)
        att2 = self.att2(x=c2, g=g2)  # (B, 128, H/2, W/2)
        concat2 = torch.cat([att2, g2], dim=1)  # (B, 256, H/2, W/2)
        d2 = self.conv_block_up2(concat2)  # (B, 128, H/2, W/2)
        
        # Decoder Block 1
        g1 = self.up1(d2)           # (B, 64, H, W)
        att1 = self.att1(x=c1, g=g1)  # (B, 64, H, W)
        concat1 = torch.cat([att1, g1], dim=1)  # (B, 128, H, W)
        d1 = self.conv_block_up1(concat1)  # (B, 64, H, W)
        
        # ==================== OUTPUT (LOGITS) ====================
        
        logits = self.output_conv(d1)  # (B, out_channels, H, W)
        
        # NO SIGMOID - return raw logits
        return logits


if __name__ == "__main__":
    # Test the Attention U-Net Logits model
    print("="*80)
    print("Testing AttentionUNetLogits")
    print("="*80)
    
    batch_size = 2
    in_channels = 1
    out_channels = 1
    height, width = 256, 256
    
    # Create model
    model = AttentionUNetLogits(
        in_channels=in_channels,
        out_channels=out_channels,
        base_filters=64
    )
    
    # Create dummy input
    x = torch.randn(batch_size, in_channels, height, width)
    
    print(f"\nInput shape: {x.shape}")
    
    # Forward pass
    logits = model(x)
    
    print(f"Output (logits) shape: {logits.shape}")
    print(f"Output range: [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Output mean: {logits.mean():.4f}")
    print(f"\nNote: Outputs are LOGITS (can be any real number)")
    print(f"      Use torch.sigmoid(logits) to get probabilities [0, 1]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Convert logits to probabilities for visualization
    probs = torch.sigmoid(logits)
    print(f"\nProbabilities (after sigmoid):")
    print(f"  Range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  Mean: {probs.mean():.4f}")
    
    print("\n" + "="*80)
    print("✓ Model test passed!")
    print("="*80)
