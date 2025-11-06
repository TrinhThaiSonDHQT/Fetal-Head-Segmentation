"""
Standard MobileNetV2-Based U-Net

A U-Net architecture where the encoder is replaced with MobileNetV2 backbone,
while the decoder uses standard U-Net blocks with channels [512, 256, 128, 64, 64].

Architecture (as per paper):
    - Encoder: MobileNetV2 backbone (pre-trained on ImageNet)
      - Extracts features at: 16, 24, 32, 96 channels (at 1/2, 1/4, 1/8, 1/16 resolution)
      - Final bottleneck: 1280 channels at 1/32 resolution
    - Decoder: Standard U-Net decoder with channels [512, 256, 128, 64, 64]
    - Skip connections: Direct concatenation from encoder to decoder

Key Features:
    - Encoder: MobileNetV2 uses depthwise separable convolutions (efficient)
    - Decoder: Standard U-Net conv blocks with [512, 256, 128, 64, 64] feature maps
    - Transfer Learning: Pre-trained weights from ImageNet
    - Efficient: Reduced parameters while maintaining accuracy

Reference:
    - MobileNetV2: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals
      and Linear Bottlenecks. CVPR 2018.
    - Paper: "Our baseline network is U-Net with input features [64, 128, 256, 512],
      and we replace its encoder with a pre-trained low computationally demanding
      model, MobileNet v2."
"""
import torch
import torch.nn as nn
import torchvision.models as models


class ConvBlock(nn.Module):
    """
    Simple convolutional block with two conv layers.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class StandardMobileNetV2UNet(nn.Module):
    """
    Standard MobileNetV2-based U-Net as per paper architecture.
    
    Architecture Overview:
        - Encoder: MobileNetV2 (pre-trained on ImageNet)
        - Decoder: Standard U-Net blocks with channels [512, 256, 128, 64, 64]
        - Skip connections: Direct concatenation from encoder to decoder
    
    MobileNetV2 Feature Extraction Points (Encoder):
        - Block 1: features[1]   -> 16 channels  @ H/2×W/2
        - Block 3: features[3]   -> 24 channels  @ H/4×W/4
        - Block 6: features[6]   -> 32 channels  @ H/8×W/8
        - Block 13: features[13] -> 96 channels  @ H/16×W/16
        - Block 18: features[18] -> 1280 channels @ H/32×W/32
    
    Decoder Channel Progression:
        - Stage 5: 1280 -> 512 (1/32 -> 1/16)
        - Stage 4: 512 -> 256 (1/16 -> 1/8)
        - Stage 3: 256 -> 128 (1/8 -> 1/4)
        - Stage 2: 128 -> 64 (1/4 -> 1/2)
        - Stage 1: 64 -> 64 (1/2 -> 1/1)
    
    Args:
        in_channels (int): Number of input channels (default=1 for grayscale)
        out_channels (int): Number of output channels (default=1 for binary segmentation)
        pretrained (bool): Use ImageNet pre-trained weights (default=True)
        freeze_encoder (bool): Freeze MobileNetV2 encoder (default=True)
    """
    
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        pretrained=True,
        freeze_encoder=True
    ):
        super(StandardMobileNetV2UNet, self).__init__()
        
        # ==================== ENCODER (MobileNetV2) ====================
        # Load pre-trained MobileNetV2
        mobilenet = models.mobilenet_v2(pretrained=pretrained)
        
        # MobileNetV2 expects 3-channel RGB input, but we have 1-channel grayscale
        # Solution: Replace first conv layer to accept 1-channel input
        if in_channels != 3:
            # Get original first conv weights (3, 32, 3, 3)
            original_conv = mobilenet.features[0][0]
            
            # Create new conv layer for grayscale input
            new_conv = nn.Conv2d(
                in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
            
            # Initialize new conv weights
            if pretrained:
                # Average RGB weights to initialize grayscale channel
                with torch.no_grad():
                    new_conv.weight[:, :, :, :] = original_conv.weight.mean(dim=1, keepdim=True)
            
            # Replace first conv layer
            mobilenet.features[0][0] = new_conv
        
        # Extract encoder features at different scales
        self.encoder = mobilenet.features
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # MobileNetV2 encoder channel counts at different stages
        # Skip connections from: [16, 24, 32, 96] at resolutions [1/2, 1/4, 1/8, 1/16]
        # Bottleneck: 1280 at resolution 1/32
        self.encoder_channels = [16, 24, 32, 96, 1280]
        
        # ==================== DECODER ====================
        # Decoder follows standard U-Net with channels [64, 128, 256, 512]
        # As per paper: "baseline network is U-Net with input features [64, 128, 256, 512]"
        
        # Decoder stage 5 (1/32 -> 1/16): 1280 -> 512
        self.up5 = nn.ConvTranspose2d(1280, 512, kernel_size=2, stride=2)
        # Input: 512 (upsampled) + 96 (skip from enc4) = 608
        self.dec5 = ConvBlock(512 + 96, 512)
        
        # Decoder stage 4 (1/16 -> 1/8): 512 -> 256
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        # Input: 256 (upsampled) + 32 (skip from enc3) = 288
        self.dec4 = ConvBlock(256 + 32, 256)
        
        # Decoder stage 3 (1/8 -> 1/4): 256 -> 128
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        # Input: 128 (upsampled) + 24 (skip from enc2) = 152
        self.dec3 = ConvBlock(128 + 24, 128)
        
        # Decoder stage 2 (1/4 -> 1/2): 128 -> 64
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        # Input: 64 (upsampled) + 16 (skip from enc1) = 80
        self.dec2 = ConvBlock(64 + 16, 64)
        
        # Decoder stage 1 (1/2 -> 1/1): 64 -> 64
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        # Input: 64 (upsampled only, no skip connection at full resolution)
        self.dec1 = ConvBlock(64, 64)
        
        # ==================== OUTPUT ====================
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Forward pass through Standard MobileNetV2 U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W)
        
        Returns:
            torch.Tensor: Output segmentation map of shape (B, C_out, H, W)
        """
        # ==================== ENCODER (MobileNetV2) ====================
        # Extract features at different scales from MobileNetV2
        # MobileNetV2 features indices: [1, 3, 6, 13, 18]
        encoder_features = []
        
        # Pass through MobileNetV2 and collect intermediate features
        for i, layer in enumerate(self.encoder):
            x = layer(x)
            # Collect features at specific indices
            if i in [1, 3, 6, 13, 18]:
                encoder_features.append(x)
        
        # Unpack encoder features
        enc1, enc2, enc3, enc4, enc5 = encoder_features
        # enc1: (B, 16, H/2, W/2)    - from MobileNetV2 layer 1
        # enc2: (B, 24, H/4, W/4)    - from MobileNetV2 layer 3
        # enc3: (B, 32, H/8, W/8)    - from MobileNetV2 layer 6
        # enc4: (B, 96, H/16, W/16)  - from MobileNetV2 layer 13
        # enc5: (B, 1280, H/32, W/32) - from MobileNetV2 layer 18 (bottleneck)
        
        # ==================== DECODER ====================
        # Decoder stage 5 (1/32 -> 1/16): 1280 -> 512
        x = self.up5(enc5)  # (B, 512, H/16, W/16)
        x = torch.cat([x, enc4], dim=1)  # (B, 608, H/16, W/16)
        x = self.dec5(x)  # (B, 512, H/16, W/16)
        
        # Decoder stage 4 (1/16 -> 1/8): 512 -> 256
        x = self.up4(x)  # (B, 256, H/8, W/8)
        x = torch.cat([x, enc3], dim=1)  # (B, 288, H/8, W/8)
        x = self.dec4(x)  # (B, 256, H/8, W/8)
        
        # Decoder stage 3 (1/8 -> 1/4): 256 -> 128
        x = self.up3(x)  # (B, 128, H/4, W/4)
        x = torch.cat([x, enc2], dim=1)  # (B, 152, H/4, W/4)
        x = self.dec3(x)  # (B, 128, H/4, W/4)
        
        # Decoder stage 2 (1/4 -> 1/2): 128 -> 64
        x = self.up2(x)  # (B, 64, H/2, W/2)
        x = torch.cat([x, enc1], dim=1)  # (B, 80, H/2, W/2)
        x = self.dec2(x)  # (B, 64, H/2, W/2)
        
        # Decoder stage 1 (1/2 -> 1/1): 64 -> 64
        x = self.up1(x)  # (B, 64, H, W)
        # No skip connection at full resolution as per paper architecture
        x = self.dec1(x)  # (B, 64, H, W)
        
        # ==================== OUTPUT ====================
        x = self.out_conv(x)  # (B, out_channels, H, W)
        x = self.sigmoid(x)
        
        return x


def count_parameters(model):
    """
    Count the total and trainable parameters in the model.
    
    Args:
        model (nn.Module): PyTorch model
    
    Returns:
        tuple: (total_params, trainable_params, frozen_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = StandardMobileNetV2UNet(
        in_channels=1,
        out_channels=1,
        pretrained=True,
        freeze_encoder=True
    ).to(device)
    
    # Test with dummy input
    dummy_input = torch.randn(2, 1, 256, 256).to(device)
    output = model(dummy_input)
    
    # Print model information
    print("=" * 80)
    print("Standard MobileNetV2 U-Net Model Summary")
    print("=" * 80)
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print("-" * 80)
    
    total_params, trainable_params, frozen_params = count_parameters(model)
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.1f}%)")
    print(f"Frozen parameters:    {frozen_params:,} ({100 * frozen_params / total_params:.1f}%)")
    print(f"Total model size:     ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    print(f"Trainable size:       ~{trainable_params * 4 / (1024**2):.2f} MB (float32)")
    print("=" * 80)
    
    # Architecture summary
    print("\nArchitecture Highlights:")
    print("  • Encoder: MobileNetV2 (pre-trained on ImageNet)")
    print("    - Extracts features at: 16, 24, 32, 96, 1280 channels")
    print("    - Resolutions: H/2, H/4, H/8, H/16, H/32")
    print("  • Decoder: Standard U-Net blocks with channels [512, 256, 128, 64, 64]")
    print("    - 5 upsampling stages: 512 -> 256 -> 128 -> 64 -> 64")
    print("  • Skip connections: Direct concatenation from encoder to decoder")
    print("\nKey Advantages:")
    print("  ✓ Efficient: MobileNetV2 encoder uses depthwise separable convolutions")
    print("  ✓ Transfer Learning: Pre-trained ImageNet weights")
    print("  ✓ Standard U-Net decoder: Proven architecture with [512, 256, 128, 64, 64] channels")
    print("  ✓ Fast Training: Frozen encoder option reduces trainable parameters")
    print("=" * 80)
