"""
Standard U-Net Implementation for Fetal Head Segmentation
Based on the original U-Net architecture (Ronneberger et al., 2015)
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Double Convolution Block: (Conv2d -> BatchNorm -> ReLU) x 2
    Uses padding='same' to maintain spatial dimensions
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """
    Downscaling Block: MaxPool -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    Upscaling Block: ConvTranspose2d -> Concatenate -> DoubleConv
    """
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Args:
            x1: Feature map from decoder path
            x2: Feature map from encoder path (skip connection)
        """
        x1 = self.up(x1)
        
        # Handle potential size mismatches (if input size is not divisible by 16)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                                     diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class StandardUNet(nn.Module):
    """
    Standard U-Net architecture for binary segmentation
    
    Architecture:
        - Encoder: 4 downsampling blocks (64, 128, 256, 512 channels)
        - Bottleneck: 1024 channels
        - Decoder: 4 upsampling blocks (512, 256, 128, 64 channels)
        - Output: 1 channel with sigmoid activation
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale)
        out_channels (int): Number of output channels (1 for binary segmentation)
        base_filters (int): Number of filters in the first layer (default: 64)
    """
    def __init__(self, in_channels=1, out_channels=1, base_filters=64):
        super(StandardUNet, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Encoder (Contracting Path)
        self.inc = DoubleConv(in_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 16)
        
        # Decoder (Expansive Path)
        self.up1 = Up(base_filters * 16, base_filters * 8)
        self.up2 = Up(base_filters * 8, base_filters * 4)
        self.up3 = Up(base_filters * 4, base_filters * 2)
        self.up4 = Up(base_filters * 2, base_filters)
        
        # Output Layer
        self.outc = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)      # 64 channels
        x2 = self.down1(x1)   # 128 channels
        x3 = self.down2(x2)   # 256 channels
        x4 = self.down3(x3)   # 512 channels
        x5 = self.down4(x4)   # 1024 channels (bottleneck)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)  # 512 channels
        x = self.up2(x, x3)   # 256 channels
        x = self.up3(x, x2)   # 128 channels
        x = self.up4(x, x1)   # 64 channels
        
        # Output
        logits = self.outc(x)
        return self.sigmoid(logits)

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_device(self):
        """Get the device the model is on"""
        return next(self.parameters()).device


if __name__ == "__main__":
    # Test the model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")
    
    model = StandardUNet(in_channels=1, out_channels=1, base_filters=64)
    model = model.to(device)
    print(f"Model device: {model.get_device()}")
    print(f"Total trainable parameters: {model.count_parameters():,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 256, 256).to(device)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Input device: {x.device}")
    print(f"Output shape: {output.shape}")
    print(f"Output device: {output.device}")
    print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
