"""
Data augmentation and transformation utilities using Albumentations.

These augmentations maintain invariant properties: sharpness, contrast, chrominance, and gray values.
Transforms are applied on-the-fly during training by the DataLoader.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


class NormalizeMask(A.ImageOnlyTransform):
    """Custom transform to normalize mask to binary [0, 1] after other transforms."""
    
    def apply(self, img, **params):
        # This is applied to the image, so we pass through unchanged
        return img
    
    def apply_to_mask(self, mask, **params):
        # Normalize mask to binary [0, 1]
        return (mask > 0).astype(np.float32)


def get_transforms(height=256, width=256, is_train=True):
    """
    Get Albumentations transforms for training or validation/testing.
    
    For training, applies geometric augmentations (horizontal flip, rotation, translation, scaling)
    that maintain invariant properties as specified in the research article.
    
    Args:
        height (int): Target height for resizing images. Default: 256
        width (int): Target width for resizing images. Default: 256
        is_train (bool): If True, returns augmentation transforms for training.
                        If False, returns only basic transforms for validation/testing.
    
    Returns:
        albumentations.Compose: Composition of transforms to be applied to images and masks.
    """
    if is_train:
        # Training transforms with data augmentation
        # Note: Augmentations maintain invariant sharpness, contrast, chrominance, and gray values
        transforms = A.Compose([
            # Resize to target dimensions (256×256)
            A.Resize(height=height, width=width),
            
            # Horizontal flipping with 50% probability
            A.HorizontalFlip(p=0.5),
            
            # Random rotations within ±20 degrees
            A.Rotate(limit=20, p=0.5),
            
            # Translations (±10%) and scaling (±10%)
            # rotate_limit=0 to avoid additional rotation (already handled above)
            A.ShiftScaleRotate(
                shift_limit=0.1,    # Translations: ±10% of image dimensions
                scale_limit=0.1,    # Scaling: ±10%
                rotate_limit=0,     # No additional rotation
                p=0.5
            ),
            
            # Normalize pixel values from [0, 255] to [0, 1] (images only, not masks)
            A.Normalize(
                mean=(0.0,),        # Single channel (grayscale)
                std=(1.0,),         # Single channel (grayscale)
                max_pixel_value=255.0
            ),
            
            # Convert to PyTorch tensor with shape (C, H, W)
            ToTensorV2()
        ])
    
    else:
        # Validation/Testing transforms without augmentation
        transforms = A.Compose([
            # Resize to target dimensions (256×256)
            A.Resize(height=height, width=width),
            
            # Normalize pixel values from [0, 255] to [0, 1] (images only, not masks)
            A.Normalize(
                mean=(0.0,),        # Single channel (grayscale)
                std=(1.0,),         # Single channel (grayscale)
                max_pixel_value=255.0
            ),
            
            # Convert to PyTorch tensor with shape (C, H, W)
            ToTensorV2()
        ])
    
    return transforms
