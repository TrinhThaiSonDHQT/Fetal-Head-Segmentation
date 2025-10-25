"""
Cached Dataset for Faster Training

Preprocesses and caches images/masks to disk to avoid redundant preprocessing.
Useful when preprocessing is expensive (resizing, normalization, etc.).
"""
import os
import pickle
from pathlib import Path
from typing import Optional, Callable
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class CachedHC18Dataset(Dataset):
    """
    HC18 Dataset with preprocessing cache for faster training.
    
    Preprocesses images/masks once and saves to disk, then loads from cache.
    
    Args:
        image_dir (str): Path to directory containing ultrasound images
        mask_dir (str): Path to directory containing segmentation masks
        cache_dir (str): Path to directory for storing cached preprocessed data
        img_height (int): Target image height (default: 256)
        img_width (int): Target image width (default: 256)
        transform (callable, optional): Albumentations transform for augmentation only
        force_rebuild (bool): If True, rebuild cache even if it exists
    """
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        cache_dir: str,
        img_height: int = 256,
        img_width: int = 256,
        transform: Optional[Callable] = None,
        force_rebuild: bool = False
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cache_dir = Path(cache_dir)
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Define cache metadata file
        self.cache_metadata_file = self.cache_dir / "metadata.pkl"
        
        # Check if cache exists and is valid
        if not force_rebuild and self._is_cache_valid():
            print(f"Loading from cache: {self.cache_dir}")
            self._load_metadata()
        else:
            print(f"Building cache: {self.cache_dir}")
            self._build_cache()
    
    def _is_cache_valid(self) -> bool:
        """Check if cache exists and metadata file is present."""
        return self.cache_metadata_file.exists()
    
    def _load_metadata(self):
        """Load metadata from cache."""
        with open(self.cache_metadata_file, 'rb') as f:
            metadata = pickle.load(f)
        
        self.image_files = metadata['image_files']
        self.num_samples = len(self.image_files)
        print(f"Loaded {self.num_samples} cached samples")
    
    def _build_cache(self):
        """Preprocess and cache all images/masks."""
        # Get list of image files
        all_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        image_files = [f for f in all_files if '_Annotation' not in f]
        
        # Build mask filenames
        mask_files = []
        valid_image_files = []
        
        for img_file in image_files:
            # Try multiple naming conventions for masks
            # 1. Try with _Annotation suffix (original HC18 format)
            mask_file_with_suffix = img_file.replace('.png', '_Annotation.png') \
                                             .replace('.jpg', '_Annotation.jpg') \
                                             .replace('.jpeg', '_Annotation.jpeg')
            
            # 2. Try with same filename as image
            mask_file_same_name = img_file
            
            # Check which mask file exists
            mask_path_with_suffix = os.path.join(self.mask_dir, mask_file_with_suffix)
            mask_path_same_name = os.path.join(self.mask_dir, mask_file_same_name)
            
            if os.path.exists(mask_path_with_suffix):
                mask_files.append(mask_file_with_suffix)
                valid_image_files.append(img_file)
            elif os.path.exists(mask_path_same_name):
                mask_files.append(mask_file_same_name)
                valid_image_files.append(img_file)
        
        self.image_files = valid_image_files
        self.num_samples = len(self.image_files)
        
        print(f"Preprocessing {self.num_samples} image-mask pairs...")
        
        # Process and cache each sample
        for idx, (img_file, mask_file) in enumerate(tqdm(
            zip(valid_image_files, mask_files),
            total=self.num_samples,
            desc="Caching"
        )):
            # Load image and mask
            img_path = os.path.join(self.image_dir, img_file)
            mask_path = os.path.join(self.mask_dir, mask_file)
            
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Preprocess: Resize
            image = cv2.resize(image, (self.img_width, self.img_height))
            mask = cv2.resize(mask, (self.img_width, self.img_height))
            
            # Normalize image to [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # Binarize mask: threshold at 0.5 after normalization
            mask = (mask.astype(np.float32) / 255.0 > 0.5).astype(np.float32)
            
            # Save to cache
            cache_file = self.cache_dir / f"sample_{idx:04d}.npz"
            np.savez_compressed(cache_file, image=image, mask=mask)
        
        # Save metadata
        metadata = {'image_files': valid_image_files}
        with open(self.cache_metadata_file, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"Cache built successfully: {self.num_samples} samples")
    
    def __len__(self) -> int:
        """Return the total number of samples."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load preprocessed sample from cache and apply augmentations.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask) as PyTorch tensors with shape (C, H, W)
        """
        # Load from cache
        cache_file = self.cache_dir / f"sample_{idx:04d}.npz"
        data = np.load(cache_file)
        
        image = data['image']  # Shape: (H, W), dtype: float32, range: [0, 1]
        mask = data['mask']    # Shape: (H, W), dtype: float32, range: {0, 1}
        
        # Apply augmentations if provided (Albumentations)
        if self.transform is not None:
            # Convert to uint8 for Albumentations (expects [0, 255])
            image_uint8 = (image * 255).astype(np.uint8)
            mask_uint8 = (mask * 255).astype(np.uint8)
            
            transformed = self.transform(image=image_uint8, mask=mask_uint8)
            image = transformed['image']  # Tensor (1, H, W)
            mask = transformed['mask']    # Tensor (H, W)
            
            # Ensure mask has channel dimension
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)  # (H, W) -> (1, H, W)
            
            # Normalize back to [0, 1] if transform didn't
            if image.max() > 1.0:
                image = image.float() / 255.0
            if mask.max() > 1.0:
                mask = mask.float() / 255.0
        else:
            # Convert to tensors manually
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
            mask = torch.from_numpy(mask).unsqueeze(0)    # (1, H, W)
        
        return image, mask
