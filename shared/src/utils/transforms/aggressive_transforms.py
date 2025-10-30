"""
AGGRESSIVE Augmentation for Medical Image Segmentation
Designed for extreme class imbalance (0.5% foreground)

This is what papers actually use but don't fully disclose.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np


def get_aggressive_transforms(height=256, width=256, is_train=True):
    """
    Get AGGRESSIVE augmentation transforms (what papers actually use).
    
    This includes:
    - Elastic deformation (critical for medical images)
    - Grid distortion
    - Gaussian noise
    - Gaussian blur
    - CLAHE (contrast enhancement)
    - Brightness/Contrast adjustment
    - Higher augmentation probabilities
    
    Args:
        height (int): Target height. Default: 256
        width (int): Target width. Default: 256
        is_train (bool): If True, returns aggressive augmentation.
    
    Returns:
        albumentations.Compose: Composition of transforms
    """
    if is_train:
        transforms = A.Compose([
            # Resize first
            A.Resize(height=height, width=width),
            
            # === GEOMETRIC AUGMENTATIONS ===
            
            # Flips (standard)
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            
            # Rotation
            A.Rotate(limit=20, p=0.7),  # INCREASED probability from 0.5
            
            # Shift and scale (using Affine instead of deprecated ShiftScaleRotate)
            A.Affine(
                translate_percent={'x': (-0.15, 0.15), 'y': (-0.15, 0.15)},  # INCREASED from 0.1
                scale=(0.85, 1.15),    # INCREASED from 0.1
                rotate=0,
                p=0.7               # INCREASED from 0.5
            ),
            
            # === ELASTIC DEFORMATION (CRITICAL FOR MEDICAL IMAGES) ===
            # Simulates tissue deformation in ultrasound
            A.ElasticTransform(
                alpha=120,
                sigma=120 * 0.05,
                p=0.5  # 50% chance - very important
            ),
            
            # === GRID DISTORTION ===
            # Creates localized warping
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.3,
                p=0.3
            ),
            
            # === OPTICAL/INTENSITY AUGMENTATIONS ===
            # (Applied to image only, not mask)
            
            # Gaussian noise (simulates sensor noise)
            A.GaussNoise(
                var_limit=(10.0, 50.0),  # Variance range
                p=0.3
            ),
            
            # Gaussian blur (simulates focus issues)
            A.GaussianBlur(
                blur_limit=(3, 7),  # Kernel size range
                p=0.2
            ),
            
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Enhances local contrast - important for ultrasound
            A.CLAHE(
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                p=0.3
            ),
            
            # Random brightness and contrast
            A.RandomBrightnessContrast(
                brightness_limit=0.2,  # ±20%
                contrast_limit=0.2,    # ±20%
                p=0.3
            ),
            
            # Normalize to [0, 1]
            A.Normalize(
                mean=(0.0,),
                std=(1.0,),
                max_pixel_value=255.0
            ),
            
            # Convert to tensor
            ToTensorV2()
        ])
    else:
        # Validation: no augmentation
        transforms = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
            ToTensorV2()
        ])
    
    return transforms


def get_medium_transforms(height=256, width=256, is_train=True):
    """
    Medium augmentation intensity (compromise between basic and aggressive).
    
    Use this if aggressive augmentation is too slow or causes issues.
    """
    if is_train:
        transforms = A.Compose([
            A.Resize(height=height, width=width),
            
            # Geometric
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.6),
            A.Affine(
                translate_percent={'x': (-0.12, 0.12), 'y': (-0.12, 0.12)},
                scale=(0.88, 1.12),
                rotate=0,
                p=0.6
            ),
            
            # Elastic only (most important)
            A.ElasticTransform(alpha=120, sigma=6, p=0.4),
            
            # Light intensity augmentation
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.2),
            
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
            ToTensorV2()
        ])
    else:
        transforms = A.Compose([
            A.Resize(height=height, width=width),
            A.Normalize(mean=(0.0,), std=(1.0,), max_pixel_value=255.0),
            ToTensorV2()
        ])
    
    return transforms


# Example usage
if __name__ == "__main__":
    print("="*80)
    print("AGGRESSIVE AUGMENTATION TEST")
    print("="*80)
    
    import cv2
    import matplotlib.pyplot as plt
    
    # Load sample image and mask
    img_path = r"e:\Fetal Head Segmentation\shared\dataset\training_set\images\000_HC.png"
    mask_path = r"e:\Fetal Head Segmentation\shared\dataset\training_set\masks\000_HC_Annotation.png"
    
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    print(f"Original image: {img.shape}, range [{img.min()}, {img.max()}]")
    print(f"Original mask: {mask.shape}, unique {np.unique(mask)}")
    
    # Test aggressive augmentation
    transform = get_aggressive_transforms(256, 256, is_train=True)
    
    # Apply multiple times to visualize variety
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(4):
        augmented = transform(image=img, mask=mask)
        aug_img = augmented['image']
        aug_mask = augmented['mask']
        
        # Convert back for visualization
        img_np = aug_img.numpy()[0]  # (1, H, W) -> (H, W)
        mask_np = aug_mask.numpy()[0]
        
        axes[i, 0].imshow(img_np, cmap='gray')
        axes[i, 0].set_title(f'Aug {i+1}: Image')
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(mask_np, cmap='gray')
        axes[i, 1].set_title(f'Aug {i+1}: Mask')
        axes[i, 1].axis('off')
        
        # Overlay
        axes[i, 2].imshow(img_np, cmap='gray')
        axes[i, 2].imshow(mask_np, cmap='Reds', alpha=0.5)
        axes[i, 2].set_title(f'Aug {i+1}: Overlay')
        axes[i, 2].axis('off')
        
        print(f"Aug {i+1}: Mask FG% = {mask_np.mean():.4%}")
    
    plt.tight_layout()
    plt.savefig('aggressive_augmentation_preview.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved preview to: aggressive_augmentation_preview.png")
    
    print("\n" + "="*80)
    print("To use in your training:")
    print("  from shared.src.utils.aggressive_transforms import get_aggressive_transforms")
    print("  train_transform = get_aggressive_transforms(256, 256, is_train=True)")
    print("="*80)
