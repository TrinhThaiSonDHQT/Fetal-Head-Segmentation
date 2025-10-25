"""
Dataset Preparation Script
--------------------------
This script processes the HC18 dataset by:
1. Creating binary masks from annotation images
2. Splitting data into train (80%), validation (5%), and test (15%) sets
3. Organizing files into the appropriate directory structure
"""

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_binary_mask(annotation_path, output_path):
    """
    Convert annotation image to binary mask.
    
    Args:
        annotation_path: Path to annotation image
        output_path: Path to save binary mask
    """
    # Read annotation image
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
    
    if annotation is None:
        print(f"Warning: Could not read {annotation_path}")
        return False
    
    # Create binary mask (assuming annotations are non-zero pixels)
    # Threshold to create binary mask: any non-zero pixel becomes 255
    _, binary_mask = cv2.threshold(annotation, 1, 255, cv2.THRESH_BINARY)
    
    # Save binary mask
    cv2.imwrite(str(output_path), binary_mask)
    return True


def collect_image_pairs(images_dir):
    """
    Collect all image-annotation pairs from the images directory.
    
    Args:
        images_dir: Path to directory containing images and annotations
        
    Returns:
        List of tuples (image_filename, annotation_filename)
    """
    images_dir = Path(images_dir)
    pairs = []
    
    # Get all image files (not annotations)
    image_files = sorted([f for f in images_dir.glob("*.png") 
                         if "_Annotation" not in f.name])
    
    for img_file in image_files:
        # Construct corresponding annotation filename
        base_name = img_file.stem  # filename without extension
        annotation_file = images_dir / f"{base_name}_Annotation.png"
        
        if annotation_file.exists():
            pairs.append((img_file.name, annotation_file.name))
        else:
            print(f"Warning: No annotation found for {img_file.name}")
    
    return pairs


def prepare_directories(base_dir):
    """
    Create directory structure for train/val/test splits.
    
    Args:
        base_dir: Base dataset directory
    """
    base_dir = Path(base_dir)
    
    splits = ["training_set", "validation_set", "test_set"]
    subdirs = ["images", "masks"]
    
    for split in splits:
        for subdir in subdirs:
            dir_path = base_dir / split / subdir
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"Created: {dir_path}")


def split_and_organize_dataset(source_dir, base_dir, train_ratio=0.80, val_ratio=0.05, test_ratio=0.15):
    """
    Split dataset and organize into train/val/test folders.
    
    Args:
        source_dir: Directory containing source images and annotations
        base_dir: Base dataset directory
        train_ratio: Proportion for training set (default: 0.80)
        val_ratio: Proportion for validation set (default: 0.05)
        test_ratio: Proportion for test set (default: 0.15)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    source_dir = Path(source_dir)
    base_dir = Path(base_dir)
    
    # Collect all image-annotation pairs
    print("\n[1/4] Collecting image-annotation pairs...")
    pairs = collect_image_pairs(source_dir)
    print(f"Found {len(pairs)} image-annotation pairs")
    
    # Prepare directory structure
    print("\n[2/4] Preparing directory structure...")
    prepare_directories(base_dir)
    
    # Split dataset
    print("\n[3/4] Splitting dataset...")
    # First split: separate test set
    train_val_pairs, test_pairs = train_test_split(
        pairs, 
        test_size=test_ratio, 
        random_state=42,
        shuffle=True
    )
    
    # Second split: separate train and validation
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    train_pairs, val_pairs = train_test_split(
        train_val_pairs,
        test_size=val_ratio_adjusted,
        random_state=42,
        shuffle=True
    )
    
    print(f"Training set: {len(train_pairs)} samples ({len(train_pairs)/len(pairs)*100:.1f}%)")
    print(f"Validation set: {len(val_pairs)} samples ({len(val_pairs)/len(pairs)*100:.1f}%)")
    print(f"Test set: {len(test_pairs)} samples ({len(test_pairs)/len(pairs)*100:.1f}%)")
    
    # Process and copy files
    print("\n[4/4] Processing and copying files...")
    
    splits = {
        "training_set": train_pairs,
        "validation_set": val_pairs,
        "test_set": test_pairs
    }
    
    for split_name, split_pairs in splits.items():
        print(f"\nProcessing {split_name}...")
        split_dir = base_dir / split_name
        
        for img_name, annotation_name in tqdm(split_pairs, desc=f"{split_name}"):
            # Source paths
            src_img = source_dir / img_name
            src_annotation = source_dir / annotation_name
            
            # Destination paths
            dst_img = split_dir / "images" / img_name
            dst_mask = split_dir / "masks" / img_name  # Use same name as image
            
            # Copy image
            shutil.copy2(src_img, dst_img)
            
            # Create and save binary mask
            create_binary_mask(src_annotation, dst_mask)
    
    print("\n" + "="*60)
    print("Dataset preparation completed successfully!")
    print("="*60)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"  Total samples: {len(pairs)}")
    print(f"  Training:   {len(train_pairs):4d} samples -> {base_dir / 'training_set'}")
    print(f"  Validation: {len(val_pairs):4d} samples -> {base_dir / 'validation_set'}")
    print(f"  Test:       {len(test_pairs):4d} samples -> {base_dir / 'test_set'}")


def verify_dataset(base_dir):
    """
    Verify the created dataset structure and content.
    
    Args:
        base_dir: Base dataset directory
    """
    base_dir = Path(base_dir)
    print("\n" + "="*60)
    print("Verification Report")
    print("="*60)
    
    splits = ["training_set", "validation_set", "test_set"]
    
    for split in splits:
        img_dir = base_dir / split / "images"
        mask_dir = base_dir / split / "masks"
        
        num_images = len(list(img_dir.glob("*.png")))
        num_masks = len(list(mask_dir.glob("*.png")))
        
        print(f"\n{split}:")
        print(f"  Images: {num_images}")
        print(f"  Masks:  {num_masks}")
        
        if num_images != num_masks:
            print(f"  ⚠ WARNING: Mismatch between images and masks!")
        else:
            print(f"  ✓ OK")
        
        # Check a sample mask
        if num_masks > 0:
            sample_mask = list(mask_dir.glob("*.png"))[0]
            mask = cv2.imread(str(sample_mask), cv2.IMREAD_GRAYSCALE)
            unique_values = np.unique(mask)
            print(f"  Sample mask unique values: {unique_values}")
            print(f"  Sample mask shape: {mask.shape}")


if __name__ == "__main__":
    # Configuration
    SOURCE_DIR = "e:/Fetal Head Segmentation/dataset/images"
    BASE_DIR = "e:/Fetal Head Segmentation/dataset"
    
    # Split ratios
    TRAIN_RATIO = 0.80  # 80%
    VAL_RATIO = 0.05    # 5%
    TEST_RATIO = 0.15   # 15%
    
    print("="*60)
    print("HC18 Dataset Preparation")
    print("="*60)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Target directory: {BASE_DIR}")
    print(f"Split ratios - Train: {TRAIN_RATIO*100}%, Val: {VAL_RATIO*100}%, Test: {TEST_RATIO*100}%")
    print("="*60)
    
    # Check if source directory exists
    if not Path(SOURCE_DIR).exists():
        print(f"\nError: Source directory does not exist: {SOURCE_DIR}")
        exit(1)
    
    # Run dataset preparation
    split_and_organize_dataset(
        source_dir=SOURCE_DIR,
        base_dir=BASE_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # Verify the dataset
    verify_dataset(BASE_DIR)
    
    print("\n✓ All done! Your dataset is ready for training.")
