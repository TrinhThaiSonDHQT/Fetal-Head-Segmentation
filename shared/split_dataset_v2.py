"""
Split dataset from /shared/dataset/images into training and validation sets
Organizes into /shared/dataset_v2 structure with 80/20 split
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Define paths
SOURCE_DIR = Path(__file__).parent / "dataset" / "images"
TARGET_DIR = Path(__file__).parent / "dataset_v2"
TRAIN_DIR = TARGET_DIR / "traning_set"  # Keep the typo to match existing structure
VAL_DIR = TARGET_DIR / "validation_set"

# Split ratio
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2

def get_image_pairs(source_dir):
    """
    Extract unique image identifiers from the source directory.
    Returns list of base filenames (without _HC.png or _HC_Annotation.png)
    """
    all_files = sorted(os.listdir(source_dir))
    
    # Get all image files (not annotations)
    image_files = [f for f in all_files if f.endswith("_HC.png") and not f.endswith("_Annotation.png")]
    
    # Extract base names (e.g., "000_HC.png" -> "000_HC")
    base_names = [f.replace(".png", "") for f in image_files]
    
    print(f"Found {len(base_names)} image pairs")
    return base_names

def create_directories():
    """Create target directory structure"""
    for subset_dir in [TRAIN_DIR, VAL_DIR]:
        (subset_dir / "images").mkdir(parents=True, exist_ok=True)
        (subset_dir / "masks").mkdir(parents=True, exist_ok=True)
    print(f"Created directory structure in {TARGET_DIR}")

def copy_files(base_names, split_type):
    """
    Copy image and mask files to appropriate directories
    
    Args:
        base_names: List of base filenames (e.g., ["000_HC", "001_HC", ...])
        split_type: "train" or "val"
    """
    target_dir = TRAIN_DIR if split_type == "train" else VAL_DIR
    
    copied_count = 0
    for base_name in base_names:
        # Source files
        img_src = SOURCE_DIR / f"{base_name}.png"
        mask_src = SOURCE_DIR / f"{base_name}_Annotation.png"
        
        # Target files
        img_dst = target_dir / "images" / f"{base_name}.png"
        mask_dst = target_dir / "masks" / f"{base_name}_Annotation.png"
        
        # Copy files
        if img_src.exists() and mask_src.exists():
            shutil.copy2(img_src, img_dst)
            shutil.copy2(mask_src, mask_dst)
            copied_count += 1
        else:
            print(f"Warning: Missing files for {base_name}")
    
    print(f"Copied {copied_count} image pairs to {split_type} set")
    return copied_count

def main():
    print("="*60)
    print("Dataset Split Tool - 80/20 Train/Val Split")
    print("="*60)
    
    # Get all image pairs
    base_names = get_image_pairs(SOURCE_DIR)
    
    # Split into train and validation (80/20)
    train_names, val_names = train_test_split(
        base_names, 
        train_size=TRAIN_RATIO,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    
    print(f"\nSplit summary:")
    print(f"  Training set: {len(train_names)} images ({len(train_names)/len(base_names)*100:.1f}%)")
    print(f"  Validation set: {len(val_names)} images ({len(val_names)/len(base_names)*100:.1f}%)")
    
    # Create directory structure
    print(f"\nCreating directory structure...")
    create_directories()
    
    # Copy files
    print(f"\nCopying files...")
    train_count = copy_files(train_names, "train")
    val_count = copy_files(val_names, "val")
    
    # Summary
    print("\n" + "="*60)
    print("Dataset split completed successfully!")
    print("="*60)
    print(f"Training set: {train_count} image pairs")
    print(f"  Location: {TRAIN_DIR}")
    print(f"Validation set: {val_count} image pairs")
    print(f"  Location: {VAL_DIR}")
    print(f"\nTotal: {train_count + val_count} image pairs processed")
    print("="*60)

if __name__ == "__main__":
    main()
