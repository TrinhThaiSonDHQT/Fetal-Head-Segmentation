"""
Residual U-Net with Squeeze-and-Excitation (SE) Mechanism

U-Net architecture using ResidualBlockSE with SE applied after:
1. Every encoder/decoder block
2. Every skip connection before concatenation
"""
import torch
import torch.nn as nn
from .residual_block import ResidualBlockSE
from .se_block import SEBlock


class ResidualSEUNet(nn.Module):
    """
    Residual U-Net with SE mechanism.
    
    Architecture:
        - Encoder: 4 downsampling stages with ResidualBlockSE (SE integrated inside each block)
        - Bottleneck: ResidualBlockSE (SE integrated inside)
        - Decoder: 4 upsampling stages with ResidualBlockSE (SE integrated inside each block)
        - SE applied to skip connections before concatenation with decoder features
    
    SE Placement:
        1. Inside each ResidualBlockSE (after Conv2-BN, before residual addition)
        2. On skip connections before concatenation with upsampled features
    
    Args:
        in_channels (int): Number of input channels (default=1 for grayscale)
        out_channels (int): Number of output channels (default=1 for binary segmentation)
        base_channels (int): Number of channels in the first layer (default=64)
        reduction_ratio (int): Reduction ratio for SE blocks (default=16)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=64, reduction_ratio=16):
        super(ResidualSEUNet, self).__init__()
        
        # Channel progression: 64 -> 128 -> 256 -> 512 -> 1024
        channels = [base_channels * (2 ** i) for i in range(5)]
        
        # ==================== ENCODER ====================
        # Initial convolution (no downsampling)
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        
        # Encoder blocks with ResidualBlockSE (SE already integrated inside)
        self.enc1 = ResidualBlockSE(channels[0], channels[0], reduction_ratio)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc2 = ResidualBlockSE(channels[0], channels[1], reduction_ratio)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc3 = ResidualBlockSE(channels[1], channels[2], reduction_ratio)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.enc4 = ResidualBlockSE(channels[2], channels[3], reduction_ratio)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ==================== BOTTLENECK ====================
        self.bottleneck = ResidualBlockSE(channels[3], channels[4], reduction_ratio)
        
        # ==================== DECODER ====================
        # Upsampling + ResidualBlockSE (SE already integrated inside)
        self.up4 = nn.ConvTranspose2d(channels[4], channels[3], kernel_size=2, stride=2)
        self.skip4_se = SEBlock(channels[3], reduction_ratio)  # SE on skip connection
        self.dec4 = ResidualBlockSE(channels[4], channels[3], reduction_ratio)  # channels[4] due to concat
        
        self.up3 = nn.ConvTranspose2d(channels[3], channels[2], kernel_size=2, stride=2)
        self.skip3_se = SEBlock(channels[2], reduction_ratio)  # SE on skip connection
        self.dec3 = ResidualBlockSE(channels[3], channels[2], reduction_ratio)  # channels[3] due to concat
        
        self.up2 = nn.ConvTranspose2d(channels[2], channels[1], kernel_size=2, stride=2)
        self.skip2_se = SEBlock(channels[1], reduction_ratio)  # SE on skip connection
        self.dec2 = ResidualBlockSE(channels[2], channels[1], reduction_ratio)  # channels[2] due to concat
        
        self.up1 = nn.ConvTranspose2d(channels[1], channels[0], kernel_size=2, stride=2)
        self.skip1_se = SEBlock(channels[0], reduction_ratio)  # SE on skip connection
        self.dec1 = ResidualBlockSE(channels[1], channels[0], reduction_ratio)  # channels[1] due to concat
        
        # ==================== OUTPUT ====================
        self.out_conv = nn.Conv2d(channels[0], out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through Residual SE U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output segmentation map of shape (B, C_out, H, W)
        """
        # ==================== ENCODER ====================
        # Initial convolution
        x = self.init_conv(x)
        
        # Encoder stage 1 (SE already inside ResidualBlockSE)
        enc1 = self.enc1(x)
        x = self.pool1(enc1)
        
        # Encoder stage 2 (SE already inside ResidualBlockSE)
        enc2 = self.enc2(x)
        x = self.pool2(enc2)
        
        # Encoder stage 3 (SE already inside ResidualBlockSE)
        enc3 = self.enc3(x)
        x = self.pool3(enc3)
        
        # Encoder stage 4 (SE already inside ResidualBlockSE)
        enc4 = self.enc4(x)
        x = self.pool4(enc4)
        
        # ==================== BOTTLENECK ====================
        x = self.bottleneck(x)  # SE already inside ResidualBlockSE
        
        # ==================== DECODER ====================
        # Decoder stage 4
        x = self.up4(x)
        enc4 = self.skip4_se(enc4)  # Apply SE to skip connection
        x = torch.cat([x, enc4], dim=1)  # Concatenate skip connection
        x = self.dec4(x)  # SE already inside ResidualBlockSE
        
        # Decoder stage 3
        x = self.up3(x)
        enc3 = self.skip3_se(enc3)  # Apply SE to skip connection
        x = torch.cat([x, enc3], dim=1)  # Concatenate skip connection
        x = self.dec3(x)  # SE already inside ResidualBlockSE
        
        # Decoder stage 2
        x = self.up2(x)
        enc2 = self.skip2_se(enc2)  # Apply SE to skip connection
        x = torch.cat([x, enc2], dim=1)  # Concatenate skip connection
        x = self.dec2(x)  # SE already inside ResidualBlockSE
        
        # Decoder stage 1
        x = self.up1(x)
        enc1 = self.skip1_se(enc1)  # Apply SE to skip connection
        x = torch.cat([x, enc1], dim=1)  # Concatenate skip connection
        x = self.dec1(x)  # SE already inside ResidualBlockSE
        
        # ==================== OUTPUT ====================
        x = self.out_conv(x)
        x = self.sigmoid(x)
        
        return x


def count_parameters(model):
    """
    Count the total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ResidualSEUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        reduction_ratio=16
    ).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    output = model(dummy_input)
    
    # Print model information
    print("=" * 60)
    print("Residual SE U-Net Model Summary")
    print("=" * 60)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("-" * 60)
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 60)
