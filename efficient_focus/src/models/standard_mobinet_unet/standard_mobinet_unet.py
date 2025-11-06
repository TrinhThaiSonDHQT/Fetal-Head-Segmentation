"""
Standard MobileNetV2-Based U-Net

A simple U-Net architecture using MobileNetV2 as the encoder backbone.
No additional techniques (no ASPP, no SE blocks, no residual connections).
Just basic encoder-decoder with skip connections.

Architecture:
    - Encoder: MobileNetV2 backbone (pre-trained on ImageNet, frozen for transfer learning)
    - Decoder: 5 upsampling stages with simple conv blocks
    - Skip connections: Direct concatenation without attention

Key Features:
    - Simple and straightforward architecture
    - Efficient: MobileNetV2 uses depthwise separable convolutions
    - Transfer Learning: Pre-trained weights from ImageNet
    - Fast Training: Frozen encoder reduces trainable parameters

Reference:
    - MobileNetV2: Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals
      and Linear Bottlenecks. CVPR 2018.
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
    Standard MobileNetV2-based U-Net without additional techniques.
    
    Architecture Overview:
        - Encoder: MobileNetV2 (pre-trained, frozen)
        - Decoder: 5 stages with simple ConvBlock
        - Skip connections: Direct concatenation
    
    MobileNetV2 Feature Extraction Points:
        - Initial Conv (custom): 32 channels  @ H×W     (full resolution, trainable)
        - Block 1: features[1]   -> 16 channels  @ H/2×W/2
        - Block 3: features[3]   -> 24 channels  @ H/4×W/4
        - Block 6: features[6]   -> 32 channels  @ H/8×W/8
        - Block 13: features[13] -> 96 channels  @ H/16×W/16
        - Block 18: features[18] -> 1280 channels @ H/32×W/32
    
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
        
        # Initial conv layer (maintains full resolution) for first skip connection
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # MobileNetV2 channel counts at different stages
        # Resolutions: [1/1, 1/2, 1/4, 1/8, 1/16, 1/32]
        self.encoder_channels = [32, 16, 24, 32, 96, 1280]
        
        # ==================== BOTTLENECK ====================
        # Simple bottleneck conv block
        self.bottleneck = ConvBlock(self.encoder_channels[5], 512)
        
        # ==================== DECODER ====================
        # Decoder channel progression - 5 stages to match encoder
        # [1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2 -> 1/1]
        decoder_channels = [256, 128, 64, 32, 32]
        
        # Decoder stage 5 (1/32 -> 1/16)
        self.up5 = nn.ConvTranspose2d(512, decoder_channels[0], kernel_size=2, stride=2)
        # Input: 256 (upsampled) + 96 (skip from enc4) = 352
        self.dec5 = ConvBlock(decoder_channels[0] + self.encoder_channels[4], decoder_channels[0])
        
        # Decoder stage 4 (1/16 -> 1/8)
        self.up4 = nn.ConvTranspose2d(decoder_channels[0], decoder_channels[1], kernel_size=2, stride=2)
        # Input: 128 (upsampled) + 32 (skip from enc3) = 160
        self.dec4 = ConvBlock(decoder_channels[1] + self.encoder_channels[3], decoder_channels[1])
        
        # Decoder stage 3 (1/8 -> 1/4)
        self.up3 = nn.ConvTranspose2d(decoder_channels[1], decoder_channels[2], kernel_size=2, stride=2)
        # Input: 64 (upsampled) + 24 (skip from enc2) = 88
        self.dec3 = ConvBlock(decoder_channels[2] + self.encoder_channels[2], decoder_channels[2])
        
        # Decoder stage 2 (1/4 -> 1/2)
        self.up2 = nn.ConvTranspose2d(decoder_channels[2], decoder_channels[3], kernel_size=2, stride=2)
        # Input: 32 (upsampled) + 16 (skip from enc1) = 48
        self.dec2 = ConvBlock(decoder_channels[3] + self.encoder_channels[1], decoder_channels[3])
        
        # Decoder stage 1 (1/2 -> 1/1)
        self.up1 = nn.ConvTranspose2d(decoder_channels[3], decoder_channels[4], kernel_size=2, stride=2)
        # Input: 32 (upsampled) + 32 (skip from enc0) = 64
        self.dec1 = ConvBlock(decoder_channels[4] + self.encoder_channels[0], decoder_channels[4])
        
        # ==================== OUTPUT ====================
        self.out_conv = nn.Conv2d(decoder_channels[4], out_channels, kernel_size=1)
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
        # Initial conv (maintains full resolution) - for first skip connection
        enc0 = self.init_conv(x)  # (B, 32, H, W)
        
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
        # enc0: (B, 32, H, W)        - from init_conv
        # enc1: (B, 16, H/2, W/2)    - from MobileNetV2 layer 1
        # enc2: (B, 24, H/4, W/4)    - from MobileNetV2 layer 3
        # enc3: (B, 32, H/8, W/8)    - from MobileNetV2 layer 6
        # enc4: (B, 96, H/16, W/16)  - from MobileNetV2 layer 13
        # enc5: (B, 1280, H/32, W/32) - from MobileNetV2 layer 18
        
        # ==================== BOTTLENECK ====================
        x = self.bottleneck(enc5)  # (B, 512, H/32, W/32)
        
        # ==================== DECODER ====================
        # Decoder stage 5 (1/32 -> 1/16)
        x = self.up5(x)  # (B, 256, H/16, W/16)
        x = torch.cat([x, enc4], dim=1)  # (B, 352, H/16, W/16)
        x = self.dec5(x)  # (B, 256, H/16, W/16)
        
        # Decoder stage 4 (1/16 -> 1/8)
        x = self.up4(x)  # (B, 128, H/8, W/8)
        x = torch.cat([x, enc3], dim=1)  # (B, 160, H/8, W/8)
        x = self.dec4(x)  # (B, 128, H/8, W/8)
        
        # Decoder stage 3 (1/8 -> 1/4)
        x = self.up3(x)  # (B, 64, H/4, W/4)
        x = torch.cat([x, enc2], dim=1)  # (B, 88, H/4, W/4)
        x = self.dec3(x)  # (B, 64, H/4, W/4)
        
        # Decoder stage 2 (1/4 -> 1/2)
        x = self.up2(x)  # (B, 32, H/2, W/2)
        x = torch.cat([x, enc1], dim=1)  # (B, 48, H/2, W/2)
        x = self.dec2(x)  # (B, 32, H/2, W/2)
        
        # Decoder stage 1 (1/2 -> 1/1)
        x = self.up1(x)  # (B, 32, H, W)
        x = torch.cat([x, enc0], dim=1)  # (B, 64, H, W)
        x = self.dec1(x)  # (B, 32, H, W)
        
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
    print("  • Encoder: MobileNetV2 (pre-trained on ImageNet, frozen)")
    print("  • Bottleneck: Simple ConvBlock")
    print("  • Decoder: 5 stages with simple ConvBlock")
    print("  • Skip connections: Direct concatenation (no attention)")
    print("\nKey Advantages:")
    print("  ✓ Simple: No complex modules, just basic encoder-decoder")
    print("  ✓ Efficient: MobileNetV2 uses depthwise separable convolutions")
    print("  ✓ Transfer Learning: Pre-trained weights from ImageNet")
    print("  ✓ Fast Training: Frozen encoder reduces trainable parameters")
    print("=" * 80)
