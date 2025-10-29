"""
Minimal reproducible test to diagnose 30% Dice issue
Tests model + loss + training loop in isolation
"""
import sys
sys.path.insert(0, r"e:\Fetal Head Segmentation")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from shared.src.data.dataset import HC18Dataset
from shared.src.utils.transforms import get_transforms
from shared.src.metrics.segmentation_metrics import dice_coefficient

# Simple U-Net for testing
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid()  # CRITICAL: Output in [0,1]
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

print("="*80)
print("MINIMAL REPRODUCIBLE TEST")
print("="*80)

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# Dataset
img_dir = r"e:\Fetal Head Segmentation\shared\dataset_v3\training_set\images"
mask_dir = r"e:\Fetal Head Segmentation\shared\dataset_v3\training_set\masks"
transform = get_transforms(256, 256, is_train=False)

dataset = HC18Dataset(img_dir, mask_dir, transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model
model = SimpleUNet().to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Loss & Optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Get one batch
images, masks = next(iter(loader))
images, masks = images.to(device), masks.to(device)

print(f"\nBatch:")
print(f"  Images: {images.shape}, range [{images.min():.3f}, {images.max():.3f}]")
print(f"  Masks: {masks.shape}, range [{masks.min():.3f}, {masks.max():.3f}]")
print(f"  Mask FG%: {masks.mean():.4%}")

# Test forward pass BEFORE training
model.eval()
with torch.no_grad():
    outputs_init = model(images)
    dice_init = dice_coefficient(outputs_init > 0.5, masks).item()
    print(f"\nBefore training:")
    print(f"  Output range: [{outputs_init.min():.4f}, {outputs_init.max():.4f}]")
    print(f"  Output mean: {outputs_init.mean():.4f}")
    print(f"  Initial Dice (random weights): {dice_init:.4f}")

# Train for 10 steps
model.train()
print(f"\nTraining for 10 steps...")
for step in range(10):
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, masks)
    loss.backward()
    optimizer.step()
    
    with torch.no_grad():
        preds = (outputs > 0.5).float()
        dice = dice_coefficient(preds, masks).item()
    
    print(f"  Step {step+1:2d}: Loss={loss.item():.6f}, Dice={dice:.4f}, "
          f"Out_mean={outputs.mean().item():.4f}")

# Final test
model.eval()
with torch.no_grad():
    outputs_final = model(images)
    preds_final = (outputs_final > 0.5).float()
    dice_final = dice_coefficient(preds_final, masks).item()
    
    print(f"\nAfter training (10 steps):")
    print(f"  Output range: [{outputs_final.min():.4f}, {outputs_final.max():.4f}]")
    print(f"  Output mean: {outputs_final.mean():.4f}")
    print(f"  Final Dice: {dice_final:.4f}")
    print(f"  Predicted FG%: {preds_final.mean():.4%}")
    print(f"  Ground truth FG%: {masks.mean():.4%}")

# Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS:")
print("="*80)
if dice_final > 0.5:
    print("✓ Model CAN learn (Dice > 50% in just 10 steps)")
    print("  → Your full training setup has a different issue")
elif dice_final > 0.1:
    print("⚠️ Model learns SLOWLY (Dice 10-50%)")
    print("  → Possible issues: LR too high/low, wrong loss weights, bad init")
else:
    print("🔴 Model CANNOT learn (Dice < 10%)")
    print("  → Critical issue: wrong loss function, output activation, or data mismatch")

if outputs_final.mean() < 0.01:
    print("\n🔴 CRITICAL: Model outputs are near ZERO")
    print("  → Model collapsed to predicting all background")
    print("  → Try: lower LR, better initialization, weighted loss")
elif outputs_final.mean() > 0.99:
    print("\n🔴 CRITICAL: Model outputs are near ONE")
    print("  → Model collapsed to predicting all foreground")

print("="*80)
