"""
Base U-Net Model Implementation

Standard U-Net architecture for image segmentation with:
- Contracting path (encoder) with max-pooling
- Expansive path (decoder) with up-convolution
- Skip connections via crop and concatenation
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two successive 3x3 convolutions followed by ReLU activation"""
    
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling with 2x2 up-convolution, concatenation, then double conv"""
    
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        # 2x2 up-convolution (transposed convolution)
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder (to be upsampled)
            x2: Feature map from encoder (skip connection)
        """
        # Upsample x1
        x1 = self.up(x1)
        
        # Crop x2 to match x1 size if needed (for edge pixel discarding)
        # Input size: (N, C, H, W)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        
        # Apply double convolution
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution to produce final output"""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class BaseUNet(nn.Module):
    """
    Base U-Net Architecture for Image Segmentation
    
    Architecture:
        - Input: (B, in_channels, H, W)
        - Contracting path: 4 down blocks with channel doubling (64→128→256→512)
        - Bottleneck: Double conv with 1024 channels
        - Expansive path: 4 up blocks with channel halving (512→256→128→64)
        - Output: (B, out_channels, H, W)
    
    Args:
        in_channels: Number of input channels (1 for grayscale, 3 for RGB)
        out_channels: Number of output channels (1 for binary segmentation)
        base_features: Number of features in first layer (default: 64)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_features=64):
        super(BaseUNet, self).__init__()
        
        # Contracting path (Encoder)
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16)
        
        # Expansive path (Decoder)
        self.up1 = Up(base_features * 16, base_features * 8)
        self.up2 = Up(base_features * 8, base_features * 4)
        self.up3 = Up(base_features * 4, base_features * 2)
        self.up4 = Up(base_features * 2, base_features)
        
        # Output convolution
        self.outc = OutConv(base_features, out_channels)
        
        # Sigmoid activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through U-Net
        
        Args:
            x: Input tensor (B, in_channels, H, W)
            
        Returns:
            Segmentation map (B, out_channels, H, W) with sigmoid activation
        """
        # Contracting path with skip connections
        x1 = self.inc(x)      # (B, 64, H, W)
        x2 = self.down1(x1)   # (B, 128, H/2, W/2)
        x3 = self.down2(x2)   # (B, 256, H/4, W/4)
        x4 = self.down3(x3)   # (B, 512, H/8, W/8)
        x5 = self.down4(x4)   # (B, 1024, H/16, W/16) - bottleneck
        
        # Expansive path with skip connections
        x = self.up1(x5, x4)  # (B, 512, H/8, W/8)
        x = self.up2(x, x3)   # (B, 256, H/4, W/4)
        x = self.up3(x, x2)   # (B, 128, H/2, W/2)
        x = self.up4(x, x1)   # (B, 64, H, W)
        
        # Output
        logits = self.outc(x) # (B, out_channels, H, W)
        output = self.sigmoid(logits)
        
        return output
    
    def get_num_parameters(self):
        """Calculate total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Testing function
if __name__ == "__main__":
    # Test with batch of grayscale images (256x256)
    model = BaseUNet(in_channels=1, out_channels=1, base_features=64)
    x = torch.randn(4, 1, 256, 256)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {model.get_num_parameters():,}")
    
    # Verify output range (should be [0, 1] due to sigmoid)
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
