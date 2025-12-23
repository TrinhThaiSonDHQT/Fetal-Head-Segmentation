# Architecture Documentation

## Improved U-Net Architecture

### Overview

The model implements an improved U-Net architecture specifically designed for fetal head segmentation in ultrasound images.

### Key Components

#### 1. Residual Blocks

- Used in both encoder and decoder paths
- Each block contains two 3×3 convolutions with BatchNorm
- Skip connections enable better gradient flow
- Reduces vanishing gradient problem in deep networks

#### 2. Feature Pyramid + Scale Attention Module (FP+SAM)

- Consists of 3 Feature Pyramid layers
- Enables multi-scale feature fusion
- Scale attention mechanism weighs different scales adaptively
- Improves detection of fetal heads at various sizes

#### 3. ASPP (Atrous Spatial Pyramid Pooling)

- Multiple parallel atrous convolutions with different dilation rates
- Captures multi-scale contextual information
- Enlarges receptive field without increasing parameters

#### 4. Encoder-Decoder Structure

- **Downsampling:** MaxPool2d (2×2)
- **Upsampling:** ConvTranspose2d (2×2)
- **Skip connections:** Concatenation between encoder and decoder

#### 5. Output Layer

- Sigmoid activation for binary segmentation
- Single channel output (segmentation mask)

### Training Configuration

- **Loss Function:** Dice Loss + Binary Cross-Entropy
- **Optimizer:** Adam (lr=0.001)
- **Learning Rate Scheduler:** ReduceLROnPlateau
- **Batch Size:** 8-16
- **Epochs:** 100
- **Image Size:** 256×256 pixels

### Data Augmentation

Using Albumentations library:

- HorizontalFlip (p=0.5)
- Rotation (±20°, p=0.5)
- Scale (±10%, p=0.5)
- Translate (±10%, p=0.5)

All augmentations are synchronized for image-mask pairs.

## Model Variants

### 1. MobiNet + ASPP + Residual + SE

- Full implementation with all advanced modules
- Best performance: DSC ≥97.81%
- Located in: `src/models/variants/mobinet_aspp_residual_se/`

### 2. Standard MobiNet U-Net

- Baseline implementation without advanced modules
- Located in: `src/models/variants/standard_mobinet_unet/`

## File Locations

- **Model Components:** `src/models/components/`
  - `residual_block.py`
  - `aspp.py`
  - `feature_pyramid.py`
  - `se_block.py`
- **Model Variants:** `src/models/variants/`

  - `mobinet_aspp_residual_se/`
  - `standard_mobinet_unet/`

- **Configurations:** `configs/models/`
  - `mobinet_aspp_residual_se.yaml`
  - `standard_mobinet_unet.yaml`
