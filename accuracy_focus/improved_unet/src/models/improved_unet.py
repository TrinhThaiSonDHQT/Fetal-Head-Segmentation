"""
Improved U-Net Model for Fetal Head Segmentation
Implements an advanced architecture with ResidualBlocks and FP+SAM modules
"""
import torch
import torch.nn as nn
from .residual_block import ResidualBlock
from .feature_pyramid import FeaturePyramidModule
from .scale_attention import ScaleAttentionModule


class ImprovedUNet(nn.Module):
    """
    Improved U-Net architecture for precise fetal head segmentation.
    
    Key features:
    - Residual blocks in encoder/decoder
    - Feature Pyramid + Scale Attention Module (FP+SAM) for enhanced feature fusion
    - Input/Output: 256x256 grayscale images
    
    Target performance: ~97% Dice Similarity Coefficient on HC18 dataset.
    
    Args:
        in_channels (int): Number of input channels (default=1 for grayscale)
        out_channels (int): Number of output channels (default=1 for binary segmentation)
        base_channels (int): Base number of channels in first encoder stage (default=64)
    """
    
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(ImprovedUNet, self).__init__()
        
        # ==================== ENCODER (Contracting Path) ====================
        
        # Encoder Stage 1: 1 -> 64 channels
        self.enc1_residual = ResidualBlock(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Stage 2: 64 -> 128 channels
        self.enc2_residual = ResidualBlock(base_channels, base_channels * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Stage 3: 128 -> 256 channels
        self.enc3_residual = ResidualBlock(base_channels * 2, base_channels * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Encoder Stage 4: 256 -> 512 channels
        self.enc4_residual = ResidualBlock(base_channels * 4, base_channels * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # ==================== BOTTLENECK ====================
        
        # Bottleneck: 512 -> 1024 channels (simple residual block)
        self.bottleneck = ResidualBlock(base_channels * 8, base_channels * 16)
        
        # ==================== DECODER (Expanding Path) ====================
        
        # Decoder Stage 1: Upsample from 1024 to 512 channels and fuse with skip connections
        self.upconv1 = nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2)
        
        # FP+SAM for decoder stage 1 (uses encoder stages 2, 3, 4)
        # Target size will be 32x32 (after first upsample from bottleneck)
        self.fp_sam1 = FeaturePyramidModule(
            in_channels_list=[base_channels * 2, base_channels * 4, base_channels * 8],
            out_channels=base_channels * 8,
            target_size=(32, 32)
        )
        # After concatenation: 512 (upsampled) + 512 (enc4 skip) + 512 (FP+SAM) = 1536 channels
        self.dec1_residual = ResidualBlock(base_channels * 24, base_channels * 8)
        
        # Decoder Stage 2: 512 -> 256 channels
        self.upconv2 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        
        # FP+SAM for decoder stage 2 (uses encoder stages 1, 2, 3)
        # Target size will be 64x64
        self.fp_sam2 = FeaturePyramidModule(
            in_channels_list=[base_channels, base_channels * 2, base_channels * 4],
            out_channels=base_channels * 4,
            target_size=(64, 64)
        )
        # After concatenation: 256 (upsampled) + 256 (enc3 skip) + 256 (FP+SAM) = 768 channels
        self.dec2_residual = ResidualBlock(base_channels * 12, base_channels * 4)
        
        # Decoder Stage 3: 256 -> 128 channels
        self.upconv3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        
        # Simple Scale Attention for decoder stage 3 (only uses encoder stage 1)
        self.sam3 = ScaleAttentionModule(base_channels, reduction_ratio=16)
        # After concatenation: 128 (upsampled) + 128 (enc2 skip) + 64 (SAM) = 320 channels
        self.dec3_residual = ResidualBlock(base_channels * 5, base_channels * 2)
        
        # Decoder Stage 4: 128 -> 64 channels (Final upsampling)
        self.upconv4 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        # After concatenation: 64 (upsampled) + 64 (enc1 skip) = 128 channels
        self.dec4_residual = ResidualBlock(base_channels * 2, base_channels)
        
        # ==================== OUTPUT LAYER ====================
        
        # Final 1x1 convolution to produce segmentation mask
        self.final_conv = nn.Conv2d(base_channels, out_channels, kernel_size=1)
        # NO SIGMOID - use BCEWithLogitsLoss instead
    
    def forward(self, x):
        """
        Forward pass through Improved U-Net.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, 1, 256, 256)
        
        Returns:
            torch.Tensor: Segmentation mask LOGITS of shape (B, 1, 256, 256) (apply sigmoid for visualization)
        """
        # ==================== ENCODER ====================
        
        # Stage 1: Input 256x256
        enc1 = self.enc1_residual(x)      # (B, 64, 256, 256)
        enc1_pooled = self.pool1(enc1)    # (B, 64, 128, 128)
        
        # Stage 2: 128x128
        enc2 = self.enc2_residual(enc1_pooled)  # (B, 128, 128, 128)
        enc2_pooled = self.pool2(enc2)          # (B, 128, 64, 64)
        
        # Stage 3: 64x64
        enc3 = self.enc3_residual(enc2_pooled)  # (B, 256, 64, 64)
        enc3_pooled = self.pool3(enc3)          # (B, 256, 32, 32)
        
        # Stage 4: 32x32
        enc4 = self.enc4_residual(enc3_pooled)  # (B, 512, 32, 32)
        enc4_pooled = self.pool4(enc4)          # (B, 512, 16, 16)
        
        # ==================== BOTTLENECK ====================
        
        # Apply residual block at bottleneck
        bottleneck = self.bottleneck(enc4_pooled)     # (B, 1024, 16, 16)
        
        # ==================== DECODER ====================
        
        # Decoder Stage 1: 16x16 -> 32x32
        dec1_up = self.upconv1(bottleneck)      # (B, 512, 32, 32)
        
        # Apply FP+SAM using encoder stages 2, 3, 4 to enhance skip connection
        # Need to match spatial dimensions for FP module
        enc2_for_fp1 = nn.functional.interpolate(enc2, size=(32, 32), mode='bilinear', align_corners=False)
        enc3_for_fp1 = enc3  # Already 32x32
        enc4_for_fp1 = enc4  # Already 32x32
        
        fp_sam1_out = self.fp_sam1([enc2_for_fp1, enc3_for_fp1, enc4_for_fp1])  # (B, 512, 32, 32)
        
        # Concatenate: upsampled decoder + direct skip (enc4) + enhanced FP+SAM
        dec1_concat = torch.cat([dec1_up, enc4, fp_sam1_out], dim=1)  # (B, 1536, 32, 32)
        dec1 = self.dec1_residual(dec1_concat)                          # (B, 512, 32, 32)
        
        # Decoder Stage 2: 32x32 -> 64x64
        dec2_up = self.upconv2(dec1)            # (B, 256, 64, 64)
        
        # Apply FP+SAM using encoder stages 1, 2, 3 to enhance skip connection
        enc1_for_fp2 = nn.functional.interpolate(enc1, size=(64, 64), mode='bilinear', align_corners=False)
        enc2_for_fp2 = enc2  # Already 64x64
        enc3_for_fp2 = enc3  # Already 64x64
        
        fp_sam2_out = self.fp_sam2([enc1_for_fp2, enc2_for_fp2, enc3_for_fp2])  # (B, 256, 64, 64)
        
        # Concatenate: upsampled decoder + direct skip (enc3) + enhanced FP+SAM
        # 256 (upsampled) + 256 (enc3 skip) + 256 (FP+SAM) = 768 channels
        dec2_concat = torch.cat([dec2_up, enc3, fp_sam2_out], dim=1)  # (B, 768, 64, 64)
        dec2 = self.dec2_residual(dec2_concat)                          # (B, 256, 64, 64)
        
        # Decoder Stage 3: 64x64 -> 128x128
        dec3_up = self.upconv3(dec2)            # (B, 128, 128, 128)
        
        # Apply Scale Attention to encoder stage 1 to enhance skip connection
        # Need to downsample enc1 to match dec3_up size (128x128)
        enc1_for_dec3 = nn.functional.interpolate(enc1, size=(128, 128), mode='bilinear', align_corners=False)
        enc1_attended = self.sam3(enc1_for_dec3)                  # (B, 64, 128, 128)
        
        # Also add direct skip from enc2 (downsample from 128x128 to 128x128, already correct size)
        enc2_for_dec3 = enc2  # Already 128x128 originally, need to match
        enc2_for_dec3 = nn.functional.interpolate(enc2, size=(128, 128), mode='bilinear', align_corners=False)
        
        # Concatenate: upsampled decoder + direct skip (enc2) + enhanced SAM(enc1)
        # 128 (upsampled) + 128 (enc2 skip) + 64 (SAM) = 320 channels
        dec3_concat = torch.cat([dec3_up, enc2_for_dec3, enc1_attended], dim=1)  # (B, 320, 128, 128)
        dec3 = self.dec3_residual(dec3_concat)                                     # (B, 128, 128, 128)
        
        # Decoder Stage 4: 128x128 -> 256x256
        dec4_up = self.upconv4(dec3)            # (B, 64, 256, 256)
        
        # Add direct skip connection from enc1 (already at 256x256)
        # Concatenate: upsampled decoder + direct skip (enc1)
        # 64 (upsampled) + 64 (enc1 skip) = 128 channels
        dec4_concat = torch.cat([dec4_up, enc1], dim=1)  # (B, 128, 256, 256)
        dec4 = self.dec4_residual(dec4_concat)            # (B, 64, 256, 256)
        
        # ==================== OUTPUT ====================
        
        # Final convolution (return logits, NOT sigmoid)
        out = self.final_conv(dec4)             # (B, 1, 256, 256) logits
        
        return out


if __name__ == "__main__":
    """
    Test the model with a sample input
    """
    # Create model
    model = ImprovedUNet(in_channels=1, out_channels=1, base_channels=64)
    
    # Create sample input (batch_size=2, channels=1, height=256, width=256)
    x = torch.randn(2, 1, 256, 256)
    
    # Forward pass
    output = model(x)
    
    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")