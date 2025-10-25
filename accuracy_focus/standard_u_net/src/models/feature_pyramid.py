"""
Feature Pyramid Module (FP) for Improved U-Net
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .scale_attention import ScaleAttentionModule


class FeaturePyramidModule(nn.Module):
    """
    Feature Pyramid Module that fuses multi-scale features from different encoder levels.
    
    According to the ablation study, 3 Feature Pyramid layers provide the best results.
    This module takes features from different encoder levels, normalizes their channels,
    upsamples them to the same spatial size, concatenates them, and applies scale attention.
    
    Args:
        in_channels_list (list of int): List of input channels for each pyramid level
                                        (e.g., [64, 128, 256] for 3 levels)
        out_channels (int): Number of output channels (uniform across pyramid)
        target_size (tuple): Target spatial size (H, W) for upsampling all features
    """
    
    def __init__(self, in_channels_list, out_channels, target_size):
        super(FeaturePyramidModule, self).__init__()
        
        self.target_size = target_size
        self.num_levels = len(in_channels_list)
        
        # 1x1 convolutions to normalize channels for each pyramid level
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for in_channels in in_channels_list
        ])
        
        # Fusion convolution after concatenation
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(self.num_levels * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Scale Attention Module
        self.scale_attention = ScaleAttentionModule(out_channels, reduction_ratio=16)
    
    def forward(self, feature_list):
        """
        Forward pass through Feature Pyramid Module.
        
        Args:
            feature_list (list of torch.Tensor): List of feature maps from different encoder levels
                                                 [feat_low, feat_mid, feat_high]
                                                 Each has shape (B, C_i, H_i, W_i)
        
        Returns:
            torch.Tensor: Refined feature map with attention applied, shape (B, out_channels, H, W)
        """
        # Normalize channels and upsample all features to target size
        upsampled_features = []
        
        for i, (feat, lateral_conv) in enumerate(zip(feature_list, self.lateral_convs)):
            # Normalize channels
            normalized_feat = lateral_conv(feat)
            
            # Upsample to target size
            if normalized_feat.shape[2:] != self.target_size:
                upsampled_feat = F.interpolate(
                    normalized_feat,
                    size=self.target_size,
                    mode='bilinear',
                    align_corners=False
                )
            else:
                upsampled_feat = normalized_feat
            
            upsampled_features.append(upsampled_feat)
        
        # Concatenate all pyramid levels
        concatenated = torch.cat(upsampled_features, dim=1)
        
        # Fusion convolution
        fused = self.fusion_conv(concatenated)
        
        # Apply scale attention
        out = self.scale_attention(fused)
        
        return out
