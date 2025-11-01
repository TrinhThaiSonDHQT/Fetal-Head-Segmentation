"""
Split dataset from /shared/dataset/images into training, validation, and test sets
Organizes into /shared/dataset_v3 structure with 80/5/15 split
INCLUDES ALL IMAGE VARIANTS (e.g., 10_HC, 10_2HC, 10_3HC, etc.)
Creates binary masks with filled white regions from annotation outlines
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random
import cv2
import numpy as np

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Define paths
SOURCE_DIR = Path(__file__).parent / "dataset" / "images"
TARGET_DIR = Path(__file__).parent / "dataset_v3"
TRAIN_DIR = TARGET_DIR / "training_set"
VAL_DIR = TARGET_DIR / "validation_set"
TEST_DIR = TARGET_DIR / "test_set"

# Split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.05
TEST_RATIO = 0.15

def get_all_image_pairs(source_dir):
    """
    Extract ALL unique image identifiers from the source directory.
    Includes all variants (e.g., 010_HC, 010_2HC, 010_3HC, etc.)
    Returns list of base filenames (without _Annotation.png suffix)
    """
    all_files = sorted(os.listdir(source_dir))
    
    # Get all image files that end with HC.png (not annotations)
    # This includes: xxx_HC.png, xxx_2HC.png, xxx_3HC.png, xxx_4HC.png, etc.
    image_files = [
        f for f in all_files 
        if f.endswith("HC.png") and not f.endswith("_Annotation.png")
    ]
    
    # Extract base names (e.g., "010_2HC.png" -> "010_2HC")
    base_names = [f.replace(".png", "") for f in image_files]
    
    print(f"Found {len(base_names)} image pairs (including all variants)")
    return base_names

def create_directories():
    """Create target directory structure"""
    for subset_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        (subset_dir / "images").mkdir(parents=True, exist_ok=True)
        (subset_dir / "masks").mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {TARGET_DIR}")

def create_filled_mask(annotation_path):
    """
    Create a binary mask with filled white region from annotation outline
    
    Args:
        annotation_path: Path to the annotation file (outline)
    
    Returns:
        Binary mask (numpy array) with white (255) filled region
    """
    # Read annotation image (grayscale)
    annotation = cv2.imread(str(annotation_path), cv2.IMREAD_GRAYSCALE)
    
    if annotation is None:
        raise ValueError(f"Could not read annotation file: {annotation_path}")
    
    # Threshold to get binary outline (in case it's not pure binary)
    _, binary_outline = cv2.threshold(annotation, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours in the outline
    contours, _ = cv2.findContours(binary_outline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create empty mask (all black)
    filled_mask = np.zeros_like(annotation)
    
    # Fill all contours with white (255)
    if len(contours) > 0:
        cv2.drawContours(filled_mask, contours, -1, 255, thickness=cv2.FILLED)
    
    return filled_mask

def copy_files(base_names, split_type):
    """
    Copy image files and create filled binary masks
    
    Args:
        base_names: List of base filenames (e.g., ["000_HC", "010_2HC", ...])
        split_type: "train", "val", or "test"
    """
    if split_type == "train":
        target_dir = TRAIN_DIR
    elif split_type == "val":
        target_dir = VAL_DIR
    else:
        target_dir = TEST_DIR
    
    copied_count = 0
    missing_files = []
    processing_errors = []
    
    for base_name in base_names:
        # Source files
        img_src = SOURCE_DIR / f"{base_name}.png"
        mask_src = SOURCE_DIR / f"{base_name}_Annotation.png"
        
        # Target files
        img_dst = target_dir / "images" / f"{base_name}.png"
        mask_dst = target_dir / "masks" / f"{base_name}_Annotation.png"
        
        # Check if source files exist
        if not (img_src.exists() and mask_src.exists()):
            missing_files.append(base_name)
            continue
        
        try:
            # Copy original image
            shutil.copy2(img_src, img_dst)
            
            # Create filled binary mask from annotation outline
            filled_mask = create_filled_mask(mask_src)
            
            # Save the filled mask
            cv2.imwrite(str(mask_dst), filled_mask)
            
            copied_count += 1
            
        except Exception as e:
            processing_errors.append((base_name, str(e)))
    
    if missing_files:
        print(f"Warning: Missing files for {len(missing_files)} samples: {missing_files[:5]}...")
    
    if processing_errors:
        print(f"Warning: Processing errors for {len(processing_errors)} samples:")
        for name, error in processing_errors[:5]:
            print(f"  - {name}: {error}")
    
    print(f"Copied {copied_count} image pairs to {split_type} set")
    return copied_count

def main():
    print("="*60)
    print("Dataset Split Tool v3 - 80/5/15 Train/Val/Test Split")
    print("Includes ALL image variants (e.g., xxx_2HC, xxx_3HC, etc.)")
    print("Creates filled binary masks from annotation outlines")
    print("="*60)
    
    # Get all image pairs (including all variants)
    base_names = get_all_image_pairs(SOURCE_DIR)
    
    # Split into train, validation, and test sets
    # First split: 80% train, 20% temp (val + test)
    train_names, temp_names = train_test_split(
        base_names, 
        train_size=TRAIN_RATIO,
        test_size=(VAL_RATIO + TEST_RATIO),
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    # Second split: from temp, split into val and test
    # val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO) = 0.05 / 0.20 = 0.25
    val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val_names, test_names = train_test_split(
        temp_names,
        train_size=val_ratio_adjusted,
        test_size=(1 - val_ratio_adjusted),
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"\nSplit summary:")
    print(f"  Training set: {len(train_names)} images ({len(train_names)/len(base_names)*100:.1f}%)")
    print(f"  Validation set: {len(val_names)} images ({len(val_names)/len(base_names)*100:.1f}%)")
    print(f"  Test set: {len(test_names)} images ({len(test_names)/len(base_names)*100:.1f}%)")
    
    # Create directory structure
    print(f"\nCreating directory structure...")
    create_directories()
    
    # Copy files
    print(f"\nCopying files...")
    train_count = copy_files(train_names, "train")
    val_count = copy_files(val_names, "val")
    test_count = copy_files(test_names, "test")
    
    # Summary
    print("\n" + "="*60)
    print("Dataset split completed successfully!")
    print("="*60)
    print(f"Training set: {train_count} image pairs")
    print(f"  Location: {TRAIN_DIR}")
    print(f"Validation set: {val_count} image pairs")
    print(f"  Location: {VAL_DIR}")
    print(f"Test set: {test_count} image pairs")
    print(f"  Location: {TEST_DIR}")
    print(f"\nTotal: {train_count + val_count + test_count} image pairs processed")
    print("="*60)

if __name__ == "__main__":
    main()
