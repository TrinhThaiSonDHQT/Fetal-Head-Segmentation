"""
HC18 Fetal Head Segmentation Dataset
"""
import os
from typing import Optional, Callable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class HC18Dataset(Dataset):
    """
    PyTorch Dataset for HC18 fetal head segmentation challenge.
    
    Loads grayscale ultrasound images and their corresponding binary segmentation masks.
    
    Args:
        image_dir (str): Path to directory containing ultrasound images
        mask_dir (str): Path to directory containing segmentation masks
        transform (callable, optional): Albumentations transform to apply to images and masks
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # Get sorted list of image filenames (excluding masks)
        all_files = sorted([
            f for f in os.listdir(image_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        # Filter to get only actual images (not annotation/mask files)
        self.image_files = [f for f in all_files if '_Annotation' not in f]
        
        # Build corresponding mask filenames
        # Masks have '_Annotation' suffix before file extension
        self.mask_files = []
        for img_file in self.image_files:
            # Create mask filename by adding '_Annotation' before extension
            # e.g., 'image_001.png' -> 'image_001_Annotation.png'
            name, ext = os.path.splitext(img_file)
            mask_file = f"{name}_Annotation{ext}"
            mask_path = os.path.join(mask_dir, mask_file)
            if os.path.exists(mask_path):
                self.mask_files.append(mask_file)
            else:
                print(f"Warning: Mask not found for image {img_file} (expected: {mask_file})")
        
        # Verify that all images have corresponding masks
        assert len(self.image_files) == len(self.mask_files), \
            f"Number of images ({len(self.image_files)}) != number of masks ({len(self.mask_files)})"
        
        print(f"Loaded {len(self.image_files)} image-mask pairs from {image_dir}")
    
    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return len(self.image_files)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load and return the image and mask at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask) as PyTorch tensors with shape (C, H, W)
                - image: Grayscale image normalized to [0, 1]
                - mask: Binary mask with values 0 or 1
        """
        # Get file paths
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        
        # Load image as grayscale (single channel)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
        
        # Load mask as grayscale
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to load mask: {mask_path}")
        
        # Apply transformations if provided (Albumentations)
        if self.transform is not None:
            # Albumentations with ToTensorV2 handles augmentation and tensorization
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']  # Tensor (C, H, W) - A.Normalize() handles [0,1] normalization
            mask = transformed['mask']    # Tensor (H, W) - NOT normalized, still [0, 255]
            
            # Image is already normalized to [0, 1] by A.Normalize() in transforms.py
            # No additional normalization needed for images
            
            # Add channel dimension to mask: (H, W) -> (1, H, W)
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            
            # Normalize mask to binary [0, 1]
            # A.Normalize() only affects images, not masks
            # ToTensorV2 converts mask to tensor but keeps [0, 255] range
            mask = mask / 255.0
            mask = (mask > 0.5).float()
        else:
            # Manual preprocessing without Albumentations
            # Resize to 256x256
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            # Normalize image to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Ensure mask is binary (0 or 1)
            mask = (mask > 0).astype(np.float32)
            
            # Add channel dimension: (H, W) -> (1, H, W)
            image = np.expand_dims(image, axis=0)
            mask = np.expand_dims(mask, axis=0)
            
            # Convert to tensors
            image = torch.from_numpy(image).float()
            mask = torch.from_numpy(mask).float()
        
        return image, mask
