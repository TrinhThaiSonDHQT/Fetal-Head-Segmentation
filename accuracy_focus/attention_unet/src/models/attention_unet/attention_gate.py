"""
Attention Gate Module for Attention U-Net
Based on: https://arxiv.org/pdf/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    """
    2D Attention Gate module from Attention U-Net paper.
    
    This module computes attention coefficients to weight skip connections,
    allowing the model to focus on relevant spatial regions.
    
    Args:
        F_g (int): Number of channels in the gating signal (g)
        F_l (int): Number of channels in the skip connection input (x)
        F_int (int): Number of intermediate filters
        stride (int): Stride for downsampling x to match g spatial dimensions (default: 2)
    """
    
    def __init__(self, F_g: int, F_l: int, F_int: int, stride: int = 2):
        super(AttentionGate, self).__init__()
        
        self.stride = stride
        
        # W_g: 1x1 convolution for gating signal (at coarse resolution)
        self.W_g = nn.Conv2d(
            in_channels=F_g,
            out_channels=F_int,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # W_x: Convolution with stride to downsample skip connection to match gating signal
        self.W_x = nn.Conv2d(
            in_channels=F_l,
            out_channels=F_int,
            kernel_size=self.stride,
            stride=self.stride,
            padding=0,
            bias=False
        )
        
        # psi: 1x1 convolution to generate attention coefficients
        self.psi = nn.Conv2d(
            in_channels=F_int,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
        
        # Activations
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Attention Gate.
        
        Args:
            x (torch.Tensor): Skip connection feature map (B, F_l, H_x, W_x)
            g (torch.Tensor): Gating signal from decoder (B, F_g, H_g, W_g)
        
        Returns:
            torch.Tensor: Attention-weighted feature map (B, F_l, H_x, W_x)
        """
        # Store original x shape for final upsampling
        original_size = x.size()[2:]  # (H_x, W_x)
        
        # Step 1: Transform g with W_g (phi_g) - at coarse resolution
        phi_g = self.W_g(g)  # (B, F_int, H_g, W_g)
        
        # Step 2: Transform and downsample x with W_x (theta_x_down)
        theta_x_down = self.W_x(x)  # (B, F_int, H_g, W_g)
        
        # Step 3: Add phi_g and theta_x_down
        add = phi_g + theta_x_down  # (B, F_int, H_g, W_g)
        
        # Step 4: Apply ReLU activation (sigma_1)
        relu_out = self.relu(add)  # (B, F_int, H_g, W_g)
        
        # Step 5: Apply psi transformation
        psi_out = self.psi(relu_out)  # (B, 1, H_g, W_g)
        
        # Step 6: Apply Sigmoid to get coarse attention coefficients (sigma_2)
        alpha_coarse = self.sigmoid(psi_out)  # (B, 1, H_g, W_g)
        
        # Step 7: Resample alpha_coarse to match original x spatial dimensions
        alpha_fine = F.interpolate(
            alpha_coarse,
            size=original_size,
            mode='bilinear',
            align_corners=True
        )  # (B, 1, H_x, W_x)
        
        # Step 8: Multiply original input x by attention coefficients alpha_fine
        # alpha_fine is broadcast across the channel dimension
        x_hat = x * alpha_fine  # (B, F_l, H_x, W_x)
        
        return x_hat


if __name__ == "__main__":
    # Test the Attention Gate module
    batch_size = 2
    F_g = 256  # Channels in gating signal
    F_l = 512  # Channels in skip connection
    F_int = 256  # Intermediate filters
    
    # Create attention gate
    attention_gate = AttentionGate(F_g=F_g, F_l=F_l, F_int=F_int)
    
    # Create dummy inputs
    x = torch.randn(batch_size, F_l, 64, 64)  # Skip connection (larger spatial size)
    g = torch.randn(batch_size, F_g, 32, 32)  # Gating signal (smaller spatial size)
    
    # Forward pass
    output = attention_gate(x, g)
    
    print(f"Input x shape: {x.shape}")
    print(f"Gating signal g shape: {g.shape}")
    print(f"Output x_hat shape: {output.shape}")
    print(f"Output shape matches input x: {output.shape == x.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in attention_gate.parameters())
    print(f"Total parameters: {total_params:,}")
