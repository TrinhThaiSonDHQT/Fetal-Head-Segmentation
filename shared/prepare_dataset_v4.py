"""
Prepare dataset_v4 from Large-Scale Annotation Dataset for Fetal Head Biometry
- 70/15/15 split for train/val/test
- Patient-level split (no data leakage)
- Stratified by anatomical plane (cerebellum/thalamic/ventricular/diverse)
"""

import os
import shutil
import random
from pathlib import Path
from collections import defaultdict
import csv

# Set random seed for reproducibility
random.seed(42)

# Source and destination paths
SOURCE_DIR = Path("e:/Fetal Head Segmentation/shared/Large-Scale Annotation Dataset for Fetal Head Biometry")
DEST_DIR = Path("e:/Fetal Head Segmentation/shared/dataset_v4")

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def extract_patient_id(filename):
    """Extract patient ID from filename"""
    # For files like "Patient00168_Plane3_1_of_3.png"
    if filename.startswith("Patient"):
        parts = filename.split("_")
        return parts[0]  # e.g., "Patient00168"
    # For files like "000_HC.png", "010_2HC.png"
    else:
        # Extract base number (before _HC or _2HC, etc.)
        base = filename.split("_")[0]
        return f"Diverse_{base}"

def get_plane_category(source_folder):
    """Determine anatomical plane category from source folder"""
    folder_name = source_folder.lower()
    if "cerebellum" in folder_name:
        return "cerebellum"
    elif "thalamic" in folder_name:
        return "thalamic"
    elif "ventricular" in folder_name:
        return "ventricular"
    else:
        return "diverse"

def collect_all_data():
    """Collect all image-mask pairs organized by patient and plane"""
    data = {
        "cerebellum": defaultdict(list),
        "thalamic": defaultdict(list),
        "ventricular": defaultdict(list),
        "diverse": defaultdict(list)
    }
    
    pixel_sizes = {}
    
    print("Collecting data from all sources...")
    
    # 1. Trans-cerebellum
    print("\n1. Processing Trans-cerebellum...")
    cerebellum_img_dir = SOURCE_DIR / "Trans-cerebellum" / "Trans-cerebellum"
    cerebellum_mask_dir = SOURCE_DIR / "Trans-cerebellum" / "Trans-cerebellum-PASCAL" / "SegmentationClass"
    cerebellum_csv = SOURCE_DIR / "Trans-cerebellum" / "Trans-cerebellum-Pixel-Size.csv"
    
    if cerebellum_csv.exists():
        with open(cerebellum_csv, 'r', encoding='utf-8-sig') as f:  # utf-8-sig removes BOM if present
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both "Pixel in mm" and "Pixel  in mm" (single/double space)
                pixel_key = 'Pixel in mm' if 'Pixel in mm' in row else 'Pixel  in mm'
                label_key = 'Label' if 'Label' in row else list(row.keys())[0]  # Fallback to first column
                pixel_sizes[row[label_key]] = row[pixel_key]
    
    if cerebellum_img_dir.exists() and cerebellum_mask_dir.exists():
        for img_file in cerebellum_img_dir.glob("*.png"):
            patient_id = extract_patient_id(img_file.name)
            mask_file = cerebellum_mask_dir / img_file.name
            if mask_file.exists():
                data["cerebellum"][patient_id].append({
                    "image": img_file,
                    "mask": mask_file,
                    "pixel_size": pixel_sizes.get(img_file.name, "unknown")
                })
    
    print(f"   Found {sum(len(v) for v in data['cerebellum'].values())} cerebellum images from {len(data['cerebellum'])} patients")
    
    # 2. Trans-thalamic
    print("\n2. Processing Trans-thalamic...")
    thalamic_img_dir = SOURCE_DIR / "Trans-thalamic" / "Trans-thalamic"
    thalamic_mask_dir = SOURCE_DIR / "Trans-thalamic" / "Trans-thalamic-PASCAL" / "SegmentationClass"
    thalamic_csv = SOURCE_DIR / "Trans-thalamic" / "Trans-Thalamic-Pixel-Size.csv"
    
    if thalamic_csv.exists():
        with open(thalamic_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both "Pixel in mm" and "Pixel  in mm" (single/double space)
                pixel_key = 'Pixel in mm' if 'Pixel in mm' in row else 'Pixel  in mm'
                label_key = 'Label' if 'Label' in row else list(row.keys())[0]
                pixel_sizes[row[label_key]] = row[pixel_key]
    
    if thalamic_img_dir.exists() and thalamic_mask_dir.exists():
        for img_file in thalamic_img_dir.glob("*.png"):
            patient_id = extract_patient_id(img_file.name)
            mask_file = thalamic_mask_dir / img_file.name
            if mask_file.exists():
                data["thalamic"][patient_id].append({
                    "image": img_file,
                    "mask": mask_file,
                    "pixel_size": pixel_sizes.get(img_file.name, "unknown")
                })
    
    print(f"   Found {sum(len(v) for v in data['thalamic'].values())} thalamic images from {len(data['thalamic'])} patients")
    
    # 3. Trans-ventricular
    print("\n3. Processing Trans-ventricular...")
    ventricular_img_dir = SOURCE_DIR / "Trans-ventricular" / "Trans-ventricular"
    ventricular_mask_dir = SOURCE_DIR / "Trans-ventricular" / "Trans-ventricular-PASCAL" / "SegmentationClass"
    ventricular_csv = SOURCE_DIR / "Trans-ventricular" / "Trans-ventricular-Pixel-Size.csv"
    
    if ventricular_csv.exists():
        with open(ventricular_csv, 'r', encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Handle both "Pixel in mm" and "Pixel  in mm" (single/double space)
                pixel_key = 'Pixel in mm' if 'Pixel in mm' in row else 'Pixel  in mm'
                label_key = 'Label' if 'Label' in row else list(row.keys())[0]
                pixel_sizes[row[label_key]] = row[pixel_key]
    
    if ventricular_img_dir.exists() and ventricular_mask_dir.exists():
        for img_file in ventricular_img_dir.glob("*.png"):
            patient_id = extract_patient_id(img_file.name)
            mask_file = ventricular_mask_dir / img_file.name
            if mask_file.exists():
                data["ventricular"][patient_id].append({
                    "image": img_file,
                    "mask": mask_file,
                    "pixel_size": pixel_sizes.get(img_file.name, "unknown")
                })
    
    print(f"   Found {sum(len(v) for v in data['ventricular'].values())} ventricular images from {len(data['ventricular'])} patients")
    
    # 4. Diverse Fetal Head Images
    print("\n4. Processing Diverse Fetal Head Images...")
    diverse_img_dir = SOURCE_DIR / "Diverse Fetal Head Images" / "Orginal_train_images_to_959_661"
    diverse_mask_dir = SOURCE_DIR / "Diverse Fetal Head Images" / "Test-Dataset-PASCAL" / "SegmentationClass"
    
    # For diverse images, we need to find matching masks
    if diverse_img_dir.exists() and diverse_mask_dir.exists():
        for img_file in diverse_img_dir.glob("*.png"):
            patient_id = extract_patient_id(img_file.name)
            # Try to find matching mask
            mask_file = diverse_mask_dir / img_file.name
            if mask_file.exists():
                data["diverse"][patient_id].append({
                    "image": img_file,
                    "mask": mask_file,
                    "pixel_size": "unknown"
                })
    
    print(f"   Found {sum(len(v) for v in data['diverse'].values())} diverse images from {len(data['diverse'])} patients")
    
    return data, pixel_sizes

def split_patients_stratified(data):
    """Split patients by plane with stratification"""
    splits = {
        "training_set": [],
        "validation_set": [],
        "test_set": []
    }
    
    total_images = 0
    
    print("\n" + "="*60)
    print("SPLITTING PATIENTS BY ANATOMICAL PLANE")
    print("="*60)
    
    for plane, patients_dict in data.items():
        if not patients_dict:
            continue
        
        # Get list of patient IDs
        patient_ids = list(patients_dict.keys())
        random.shuffle(patient_ids)
        
        # Calculate split indices
        n_patients = len(patient_ids)
        n_train = int(n_patients * TRAIN_RATIO)
        n_val = int(n_patients * VAL_RATIO)
        
        # Split patient IDs
        train_patients = patient_ids[:n_train]
        val_patients = patient_ids[n_train:n_train + n_val]
        test_patients = patient_ids[n_train + n_val:]
        
        # Count images per split
        train_imgs = sum(len(patients_dict[p]) for p in train_patients)
        val_imgs = sum(len(patients_dict[p]) for p in val_patients)
        test_imgs = sum(len(patients_dict[p]) for p in test_patients)
        plane_total = train_imgs + val_imgs + test_imgs
        total_images += plane_total
        
        print(f"\n{plane.upper()}:")
        print(f"  Total patients: {n_patients}")
        print(f"  Total images: {plane_total}")
        print(f"  Train: {len(train_patients)} patients ({train_imgs} images, {train_imgs/plane_total*100:.1f}%)")
        print(f"  Val:   {len(val_patients)} patients ({val_imgs} images, {val_imgs/plane_total*100:.1f}%)")
        print(f"  Test:  {len(test_patients)} patients ({test_imgs} images, {test_imgs/plane_total*100:.1f}%)")
        
        # Add to splits
        for patient_id in train_patients:
            splits["training_set"].extend([(plane, patient_id, item) for item in patients_dict[patient_id]])
        
        for patient_id in val_patients:
            splits["validation_set"].extend([(plane, patient_id, item) for item in patients_dict[patient_id]])
        
        for patient_id in test_patients:
            splits["test_set"].extend([(plane, patient_id, item) for item in patients_dict[patient_id]])
    
    print("\n" + "="*60)
    print("OVERALL SPLIT SUMMARY")
    print("="*60)
    print(f"Total images: {total_images}")
    print(f"Training:   {len(splits['training_set'])} images ({len(splits['training_set'])/total_images*100:.1f}%)")
    print(f"Validation: {len(splits['validation_set'])} images ({len(splits['validation_set'])/total_images*100:.1f}%)")
    print(f"Test:       {len(splits['test_set'])} images ({len(splits['test_set'])/total_images*100:.1f}%)")
    print("="*60)
    
    return splits

def copy_files_to_destination(splits):
    """Copy images and masks to destination directories"""
    print("\n" + "="*60)
    print("COPYING FILES TO DESTINATION")
    print("="*60)
    
    # Create destination directories
    for split_name in ["training_set", "validation_set", "test_set"]:
        for subdir in ["images", "masks"]:
            dest_path = DEST_DIR / split_name / subdir
            dest_path.mkdir(parents=True, exist_ok=True)
    
    # Copy files
    for split_name, items in splits.items():
        print(f"\n{split_name.upper().replace('_', ' ')}:")
        
        images_dest = DEST_DIR / split_name / "images"
        masks_dest = DEST_DIR / split_name / "masks"
        
        for idx, (plane, patient_id, item) in enumerate(items, 1):
            # Copy image
            img_src = item["image"]
            img_dest = images_dest / img_src.name
            shutil.copy2(img_src, img_dest)
            
            # Copy mask
            mask_src = item["mask"]
            mask_dest = masks_dest / mask_src.name
            shutil.copy2(mask_src, mask_dest)
            
            if idx % 500 == 0:
                print(f"  Copied {idx}/{len(items)} files...")
        
        print(f"  ✓ Completed: {len(items)} image-mask pairs")

def create_metadata_file(splits, pixel_sizes):
    """Create metadata CSV file with dataset information"""
    metadata_file = DEST_DIR / "dataset_metadata.csv"
    
    print(f"\nCreating metadata file: {metadata_file}")
    
    with open(metadata_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'split', 'plane', 'patient_id', 'pixel_size_mm'])
        
        for split_name, items in splits.items():
            for plane, patient_id, item in items:
                filename = item["image"].name
                pixel_size = item["pixel_size"]
                writer.writerow([filename, split_name, plane, patient_id, pixel_size])
    
    print(f"  ✓ Metadata file created")

def create_readme():
    """Create README file with dataset information"""
    readme_file = DEST_DIR / "README.md"
    
    readme_content = """# Dataset V4: Large-Scale Fetal Head Biometry Dataset

## Overview
This dataset contains fetal head ultrasound images with multi-class segmentation masks.

## Dataset Split
- **Training Set**: 70% of patients
- **Validation Set**: 15% of patients  
- **Test Set**: 15% of patients

## Split Strategy
- **Patient-level split**: All images from the same patient are in the same split (no data leakage)
- **Stratified by plane**: Each split maintains similar ratios of anatomical planes

## Anatomical Planes
1. **Trans-cerebellum**: Cerebellum plane images
2. **Trans-thalamic**: Thalamic plane images
3. **Trans-ventricular**: Ventricular plane images
4. **Diverse**: Mixed fetal head images

## Segmentation Classes
Based on PASCAL VOC format with the following labels:
- **Background** (RGB: 0,0,0)
- **Brain** (RGB: 128,0,0)
- **CSP** - Cavum Septum Pellucidum (RGB: 0,128,0)
- **LV** - Lateral Ventricles (RGB: 128,128,0)

## Directory Structure
```
dataset_v4/
├── training_set/
│   ├── images/
│   └── masks/
├── validation_set/
│   ├── images/
│   └── masks/
├── test_set/
│   ├── images/
│   └── masks/
├── dataset_metadata.csv
└── README.md
```

## Metadata File
`dataset_metadata.csv` contains:
- `filename`: Image filename
- `split`: Dataset split (training_set/validation_set/test_set)
- `plane`: Anatomical plane category
- `patient_id`: Patient identifier
- `pixel_size_mm`: Real-world pixel size in millimeters

## Source
Large-Scale Annotation Dataset for Fetal Head Biometry

## Notes
- Images are in PNG format
- Masks use PASCAL VOC color encoding
- Pixel size calibration data included for accurate biometry measurements
- Random seed: 42 (for reproducibility)
"""
    
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n  ✓ README.md created")

def main():
    """Main execution function"""
    print("="*60)
    print("PREPARING DATASET V4")
    print("="*60)
    print(f"Source: {SOURCE_DIR}")
    print(f"Destination: {DEST_DIR}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}")
    print("="*60)
    
    # Step 1: Collect all data
    data, pixel_sizes = collect_all_data()
    
    # Step 2: Split patients stratified by plane
    splits = split_patients_stratified(data)
    
    # Step 3: Copy files to destination
    copy_files_to_destination(splits)
    
    # Step 4: Create metadata file
    create_metadata_file(splits, pixel_sizes)
    
    # Step 5: Create README
    create_readme()
    
    print("\n" + "="*60)
    print("DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"\nDataset location: {DEST_DIR}")
    print("\nNext steps:")
    print("1. Review the dataset_metadata.csv file")
    print("2. Check the distribution across splits")
    print("3. Update your data loader to use the new dataset structure")
    print("="*60)

if __name__ == "__main__":
    main()
