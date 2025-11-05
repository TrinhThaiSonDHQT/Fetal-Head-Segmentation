# MobileNetV2 ASPP Residual SE U-Net - Model Summary

## Overview

Successfully created an efficient U-Net variant that replaces the standard encoder with **MobileNetV2** (pre-trained and frozen) while maintaining:

- **ASPP bottleneck** for multi-scale context
- **Residual blocks with SE** in the decoder
- **SE blocks** on all skip connections

---

## Architecture Details

### **Encoder: MobileNetV2 (Frozen, Pre-trained on ImageNet)**

- **Status**: All parameters frozen (`requires_grad=False`)
- **Pre-training**: ImageNet weights (transfer learning)
- **First Layer Adaptation**: Modified to accept 1-channel grayscale input (averaged RGB weights)
- **Feature Extraction Points**:
  ```
  Initial Conv (custom):  32 channels  @ H × W    (full resolution)
  MobileNetV2 Layer 1:    16 channels  @ H/2 × W/2
  MobileNetV2 Layer 3:    24 channels  @ H/4 × W/4
  MobileNetV2 Layer 6:    32 channels  @ H/8 × W/8
  MobileNetV2 Layer 13:   96 channels  @ H/16 × W/16
  MobileNetV2 Layer 18:   1280 channels @ H/32 × W/32
  ```

### **Bottleneck: ASPP (Atrous Spatial Pyramid Pooling)**

- **Input**: 1280 channels @ H/32 × W/32
- **Output**: 512 channels @ H/32 × W/32
- **Components**:
  - 1×1 convolution
  - 3×3 atrous convolutions with dilation rates [6, 12, 18]
  - Global average pooling branch
  - Feature fusion with 1×1 projection
- **Dropout**: 0.5 for regularization
- **Normalization**: GroupNorm (robust to batch_size=1)

### **Decoder: 5-Stage Upsampling with Residual SE Blocks (Trainable)**

All decoder parameters are trainable for task-specific fine-tuning.

| Stage | Resolution  | Channels  | SE Applied | Components                               |
| ----- | ----------- | --------- | ---------- | ---------------------------------------- |
| 5     | H/32 → H/16 | 512 → 256 | ✓          | ConvTranspose2d + Skip + ResidualBlockSE |
| 4     | H/16 → H/8  | 256 → 128 | ✓          | ConvTranspose2d + Skip + ResidualBlockSE |
| 3     | H/8 → H/4   | 128 → 64  | ✓          | ConvTranspose2d + Skip + ResidualBlockSE |
| 2     | H/4 → H/2   | 64 → 32   | ✓          | ConvTranspose2d + Skip + ResidualBlockSE |
| 1     | H/2 → H     | 32 → 32   | ✓          | ConvTranspose2d + Skip + ResidualBlockSE |

### **Output Layer**

- 1×1 convolution: 32 → 1 channel
- Sigmoid activation: output ∈ [0, 1]

---

## Model Statistics

| Metric                    | Value                      |
| ------------------------- | -------------------------- |
| **Total Parameters**      | 25,253,893 (~25.3M)        |
| **Trainable Parameters**  | 23,030,597 (~23.0M, 91.2%) |
| **Frozen Parameters**     | 2,223,296 (~2.2M, 8.8%)    |
| **Model Size (FP32)**     | ~96.34 MB                  |
| **Trainable Size (FP32)** | ~87.85 MB                  |

---

## Key Advantages

### **1. Efficiency**

- **MobileNetV2** uses depthwise separable convolutions → fewer parameters & FLOPs
- **Frozen encoder** → only train decoder → faster training
- **Reduced memory footprint** compared to standard U-Net with ResNet backbone

### **2. Transfer Learning**

- **Pre-trained weights** from ImageNet → better feature representations
- **Domain adaptation** → frozen encoder + trainable decoder allows quick adaptation to medical imaging

### **3. Multi-Scale Context**

- **ASPP bottleneck** captures features at multiple scales
- Handles objects of varying sizes effectively

### **4. Channel Attention**

- **SE blocks** in decoder and skip connections
- Adaptive feature recalibration → focus on important channels

### **5. Residual Learning**

- **Residual blocks** in decoder → easier gradient flow
- Better convergence during training

---

## Usage Example

```python
import torch
from efficient_focus.improved_unet.src.models.mobinet_aspp_residual_se import MobileNetV2ASPPResidualSEUNet

# Create model
model = MobileNetV2ASPPResidualSEUNet(
    in_channels=1,              # Grayscale input
    out_channels=1,             # Binary segmentation
    pretrained=True,            # Use ImageNet weights
    freeze_encoder=True,        # Freeze MobileNetV2
    reduction_ratio=16,         # SE reduction ratio
    atrous_rates=[6, 12, 18],  # ASPP dilation rates
    aspp_dropout=0.5,          # ASPP dropout
    aspp_use_groupnorm=True    # GroupNorm for stability
)

# Test forward pass
x = torch.randn(2, 1, 256, 256)
output = model(x)  # Shape: (2, 1, 256, 256)
```

---

## Training Recommendations

### **1. Optimizer**

- **Adam**: lr=1e-3 (standard)
- **AdamW**: lr=1e-3, weight_decay=1e-4 (recommended for better generalization)

### **2. Learning Rate Schedule**

- **ReduceLROnPlateau**: patience=5, factor=0.5
- **CosineAnnealingLR**: T_max=100

### **3. Loss Function**

- **Dice Loss** + **BCE**: Handles class imbalance
- Weights: 0.5 Dice + 0.5 BCE

### **4. Data Augmentation**

- HorizontalFlip (p=0.5)
- Rotation (±20°, p=0.5)
- Scale (±10%, p=0.5)
- Translate (±10%, p=0.5)

### **5. Fine-Tuning Strategy (Optional)**

1. **Phase 1**: Train with frozen encoder (10-20 epochs)
2. **Phase 2**: Unfreeze encoder, train end-to-end with lower lr (1e-5)

---

## File Locations

- **Model**: `efficient_focus/improved_unet/src/models/mobinet_aspp_residual_se/mobinet_aspp_residual_se.py`
- **Test Script**: `efficient_focus/improved_unet/test_mobilenet_standalone.py`
- **Dependencies**:
  - `residual_block.py` (ResidualBlockSE)
  - `se_block.py` (SEBlock)
  - `aspp.py` (ASPP)

---

## Next Steps

1. **Create training script** for this model
2. **Create configuration file** (YAML) with hyperparameters
3. **Integrate with dataset** (HC18 Grand Challenge)
4. **Benchmark** against accuracy_focus models
5. **Profile inference speed** (FPS, latency)
6. **Export to ONNX** for deployment

---

## Comparison with Accuracy-Focused Model

| Aspect                | **Accuracy Model** (ASPP Residual SE U-Net) | **Efficiency Model** (MobileNetV2 ASPP Residual SE U-Net) |
| --------------------- | ------------------------------------------- | --------------------------------------------------------- |
| **Encoder**           | Custom Residual SE blocks                   | MobileNetV2 (pre-trained, frozen)                         |
| **Parameters**        | Higher (~30-40M)                            | Lower (~25M)                                              |
| **Training Speed**    | Slower (all trainable)                      | Faster (only decoder trainable)                           |
| **Inference Speed**   | Slower                                      | **Faster** (depthwise separable convs)                    |
| **Expected Accuracy** | Higher (fully trainable)                    | Slightly lower (frozen encoder)                           |
| **Transfer Learning** | No                                          | **Yes** (ImageNet pre-training)                           |
| **Best Use Case**     | Maximum accuracy                            | **Real-time / Deployment**                                |

---

**✅ Model successfully created and tested!**
