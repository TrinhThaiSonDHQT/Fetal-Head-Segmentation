"""Quick script to debug MobileNetV2 feature extraction indices"""
import torch
import torchvision.models as models

# Load MobileNetV2
mobilenet = models.mobilenet_v2(pretrained=False)

print("MobileNetV2 Architecture:")
print("="*80)

# Test with dummy input
x = torch.randn(1, 3, 256, 256)
print(f"Input shape: {x.shape}\n")

# Pass through each layer and print shape
for i, layer in enumerate(mobilenet.features):
    x = layer(x)
    print(f"Layer {i:2d}: {x.shape} | {layer.__class__.__name__}")

print("\n" + "="*80)
print("\nTarget feature extraction points:")
print("  H/2  (128x128)")
print("  H/4  (64x64)")  
print("  H/8  (32x32)")
print("  H/16 (16x16)")
print("  H/32 (8x8)")
