"""
Comprehensive diagnosis for 30% Dice score issue
Check multiple potential problems
"""
import sys
sys.path.insert(0, r"e:\Fetal Head Segmentation")

import torch
import cv2
import numpy as np
from pathlib import Path
from shared.src.data.dataset import HC18Dataset
from shared.src.utils.transforms import get_transforms

print("="*80)
print("COMPREHENSIVE DIAGNOSIS: WHY DICE SCORE IS STUCK AT 30%?")
print("="*80)

# Dataset paths
img_dir = r"e:\Fetal Head Segmentation\shared\dataset\training_set\images"
mask_dir = r"e:\Fetal Head Segmentation\shared\dataset\training_set\masks"

# 1. Check dataset loading
print("\n[1] DATASET LOADING CHECK")
print("-" * 80)
train_transform = get_transforms(height=256, width=256, is_train=False)  # No aug for testing
dataset = HC18Dataset(img_dir, mask_dir, transform=train_transform)

img, mask = dataset[0]
print(f"✓ Dataset loads successfully")
print(f"  Image shape: {img.shape}, Mask shape: {mask.shape}")
print(f"  Image range: [{img.min():.4f}, {img.max():.4f}]")
print(f"  Mask range: [{mask.min():.4f}, {mask.max():.4f}]")
print(f"  Mask unique: {torch.unique(mask).tolist()}")
print(f"  Mask mean (FG%): {mask.mean():.4%}")

# 2. Check if mask is correctly binary
is_binary = torch.all((mask == 0) | (mask == 1)).item()
if not is_binary:
    print(f"  ⚠️ WARNING: Mask is NOT binary! Has values: {torch.unique(mask).tolist()}")
else:
    print(f"  ✓ Mask is correctly binary (0/1)")

# 3. Check multiple samples
print("\n[2] MULTIPLE SAMPLE CHECK (First 20 samples)")
print("-" * 80)
issues = []
for i in range(min(20, len(dataset))):
    img, mask = dataset[i]
    fg_ratio = mask.mean().item()
    is_bin = torch.all((mask == 0) | (mask == 1)).item()
    
    # Typical fetal head: 2-10% of image
    if fg_ratio < 0.001 or fg_ratio > 0.20:
        issues.append(f"Sample {i}: Unusual FG ratio {fg_ratio:.4%}")
    if not is_bin:
        issues.append(f"Sample {i}: Not binary")
    
    status = "✓" if (0.001 < fg_ratio < 0.20 and is_bin) else "⚠️"
    print(f"  {status} Sample {i:2d}: FG={fg_ratio:.5%}, Binary={is_bin}")

if issues:
    print(f"\n⚠️ Found {len(issues)} potential issues:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("\n✓ All samples look normal")

# 4. Simulate a prediction and calculate Dice
print("\n[3] DICE CALCULATION TEST")
print("-" * 80)

# Create a fake prediction (all zeros = predict background)
pred_all_zeros = torch.zeros_like(mask)

# Calculate Dice manually
def calc_dice(pred, target):
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

dice_all_zeros = calc_dice(pred_all_zeros, mask)
print(f"Dice (predict all background): {dice_all_zeros:.4f}")

# Create a fake prediction (all ones = predict all foreground)
pred_all_ones = torch.ones_like(mask)
dice_all_ones = calc_dice(pred_all_ones, mask)
print(f"Dice (predict all foreground): {dice_all_ones:.4f}")

# Create a random prediction
torch.manual_seed(42)
pred_random = (torch.rand_like(mask) > 0.5).float()
dice_random = calc_dice(pred_random, mask)
print(f"Dice (random prediction): {dice_random:.4f}")

# 5. Check raw image/mask files
print("\n[4] RAW FILE CHECK (Before any transforms)")
print("-" * 80)
raw_img_path = Path(img_dir) / "000_HC.png"
raw_mask_path = Path(mask_dir) / "000_HC_Annotation.png"

raw_img = cv2.imread(str(raw_img_path), cv2.IMREAD_GRAYSCALE)
raw_mask = cv2.imread(str(raw_mask_path), cv2.IMREAD_GRAYSCALE)

print(f"Raw Image:")
print(f"  Path: {raw_img_path.name}")
print(f"  Shape: {raw_img.shape}")
print(f"  Range: [{raw_img.min()}, {raw_img.max()}]")
print(f"  Mean: {raw_img.mean():.2f}")

print(f"Raw Mask:")
print(f"  Path: {raw_mask_path.name}")
print(f"  Shape: {raw_mask.shape}")
print(f"  Unique values: {np.unique(raw_mask)}")
print(f"  FG pixels: {(raw_mask > 0).sum()} / {raw_mask.size} = {(raw_mask > 0).sum() / raw_mask.size:.4%}")

# Check if mask is actually annotated (not blank)
if raw_mask.max() == 0:
    print("  🔴 CRITICAL: Mask is completely black (no annotations)!")
elif raw_mask.max() == 255:
    print("  ✓ Mask contains annotations (white pixels)")

# 6. Test augmentation impact
print("\n[5] AUGMENTATION SYNC CHECK")
print("-" * 80)
train_aug = get_transforms(height=256, width=256, is_train=True)
dataset_aug = HC18Dataset(img_dir, mask_dir, transform=train_aug)

# Load same sample multiple times with augmentation
print("Testing augmentation consistency (5 loads of same sample):")
for i in range(5):
    img_aug, mask_aug = dataset_aug[0]
    fg = mask_aug.mean().item()
    print(f"  Load {i+1}: FG% = {fg:.4%}, Binary = {torch.all((mask_aug == 0) | (mask_aug == 1)).item()}")

print("\n" + "="*80)
print("DIAGNOSIS SUMMARY")
print("="*80)

# Analyze what 30% Dice means
print("\n30% Dice score typically indicates:")
print("  1. Model predicts mostly BACKGROUND (all zeros)")
print("  2. Model predicts mostly FOREGROUND (all ones)")
print("  3. Model learns INVERSE of ground truth")
print("  4. Masks/images are MISMATCHED or CORRUPTED")
print("  5. Loss function has WRONG sign or implementation")
print(f"\nBased on test above:")
print(f"  - All background → Dice = {dice_all_zeros:.4f}")
print(f"  - All foreground → Dice = {dice_all_ones:.4f}")
print(f"  - Random pred → Dice = {dice_random:.4f}")

print("\n" + "="*80)
