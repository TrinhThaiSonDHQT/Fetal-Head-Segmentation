"""
ASPP-Enhanced Residual SE U-Net Module

This module provides the ASPP-Enhanced Residual SE U-Net model for medical
image segmentation, specifically designed for fetal head segmentation in
ultrasound images.
"""
from .aspp_residual_se_unet_model import ASPPResidualSEUNet, count_parameters

__all__ = ['ASPPResidualSEUNet', 'count_parameters']
