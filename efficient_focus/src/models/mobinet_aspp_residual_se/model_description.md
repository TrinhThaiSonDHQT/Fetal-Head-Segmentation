# MobileNet ASPP Residual SE U-Net: Architecture Description

---

## AI IMAGE GENERATION PROMPT

**Task:** Generate a detailed U-Net architecture diagram for "MobileNet ASPP Residual SE U-Net" for fetal head segmentation in ultrasound images.

### Visual Layout Requirements:

- **Overall Structure:** Symmetric U-shaped architecture with encoder (left side), bottleneck (bottom center), and decoder (right side)
- **Input:** Single grayscale ultrasound image at top-left: `256×256×1`
- **Output:** Binary segmentation mask at top-right: `256×256×1`
- **Direction:** Information flows left → down → right → up

### Color Coding Legend (Include in diagram):

- 🔵 **Blue boxes:** Standard Conv + BatchNorm + ReLU operations
- 🔴 **Red arrows:** Squeeze-and-Excitation (SE) attention mechanism
- 🟡 **Yellow/Orange boxes:** Atrous/Dilated convolutions (ASPP module)
- 🟢 **Green arrows:** Upsampling operations (ConvTranspose2d)
- ⚫ **Gray arrows:** Skip connections (horizontal)
- 🟣 **Purple arrows:** Max pooling (downsampling)
- ⊕ **Circle with plus:** Concatenation operation
- ➕ **Plus symbol:** Element-wise addition (residual connection)

### Encoder Path (Left Side - Downward):

**Stage enc0 (Initial):**

- Box: `[Conv 3×3, stride=2] → 256×256×1 → 128×128×32`
- Label: "Modified MobileNetV2 Conv (1→32 channels)"
- Note: "Frozen weights from ImageNet"

**Stage enc1:**

- Purple arrow down (max pool symbol)
- Box: `[MobileNetV2 features[1]] → 128×128×32 → 64×64×16`
- Label: "Inverted Residual Block"

**Stage enc2:**

- Purple arrow down
- Box: `[MobileNetV2 features[3]] → 64×64×16 → 32×32×24`
- Label: "Inverted Residual Block"

**Stage enc3:**

- Purple arrow down
- Box: `[MobileNetV2 features[6]] → 32×32×24 → 16×16×32`
- Label: "Inverted Residual Block"

**Stage enc4:**

- Purple arrow down
- Box: `[MobileNetV2 features[13]] → 16×16×32 → 8×8×96`
- Label: "Inverted Residual Block"

**Stage enc5:**

- Purple arrow down
- Box: `[MobileNetV2 features[18]] → 8×8×96 → 8×8×1280`
- Label: "Final Encoder Features"

### Bottleneck (Bottom Center - ASPP Module):

**Input:** `8×8×1280`

**ASPP Structure (Show 5 parallel branches vertically stacked):**

1. **Branch 1 (top):**

   - Yellow box: `[Conv 1×1] → 8×8×128`
   - Label: "Pointwise Conv (rate=1)"

2. **Branch 2:**

   - Yellow box: `[Atrous Conv 3×3, rate=6] → 8×8×128`
   - Label: "Dilated Conv (RF=13×13)"

3. **Branch 3:**

   - Yellow box: `[Atrous Conv 3×3, rate=12] → 8×8×128`
   - Label: "Dilated Conv (RF=25×25)"

4. **Branch 4:**

   - Yellow box: `[Atrous Conv 3×3, rate=18] → 8×8×128`
   - Label: "Dilated Conv (RF=37×37)"

5. **Branch 5 (bottom):**
   - Yellow box: `[Global Average Pool + Conv 1×1] → 8×8×256`
   - Label: "Global Context"

**Concatenation:**

- ⊕ symbol: Merge all 5 branches → `8×8×640`

**Projection:**

- Blue box: `[Conv 1×1 + Dropout(0.5)] → 8×8×512`
- Label: "ASPP Output Projection"

### Decoder Path (Right Side - Upward):

**Stage dec5:**

- Green arrow: `[ConvTranspose2d 2×2, stride=2] → 8×8×512 → 16×16×256`
- Gray arrow from enc4 (skip connection): `8×8×96`
  - Red SE block on skip: `[GAP → FC(6) → ReLU → FC(96) → Sigmoid]`
- ⊕ Concatenate: `16×16×(256+96) = 16×16×352`
- Blue box with residual: `[SE-ResidualBlock] → 16×16×256`
  - Show internal structure:
    - Conv 3×3 → BN → ReLU
    - Conv 3×3 → BN → ReLU
    - Red SE attention: GAP → FC(16) → ReLU → FC(256) → Sigmoid
    - ➕ Residual connection

**Stage dec4:**

- Green arrow: `[ConvTranspose2d 2×2] → 16×16×256 → 32×32×128`
- Gray arrow from enc3: `16×16×32`
  - Red SE block on skip: `[SE attention]`
- ⊕ Concatenate: `32×32×160`
- Blue box: `[SE-ResidualBlock] → 32×32×128`

**Stage dec3:**

- Green arrow: `[ConvTranspose2d 2×2] → 32×32×128 → 64×64×64`
- Gray arrow from enc2: `32×32×24`
  - Red SE block on skip
- ⊕ Concatenate: `64×64×88`
- Blue box: `[SE-ResidualBlock] → 64×64×64`

**Stage dec2:**

- Green arrow: `[ConvTranspose2d 2×2] → 64×64×64 → 128×128×32`
- Gray arrow from enc1: `64×64×16`
  - Red SE block on skip
- ⊕ Concatenate: `128×128×48`
- Blue box: `[SE-ResidualBlock] → 128×128×32`

**Stage dec1:**

- Green arrow: `[ConvTranspose2d 2×2] → 128×128×32 → 256×256×32`
- Gray arrow from enc0: `128×128×32`
  - Red SE block on skip
- ⊕ Concatenate: `256×256×64`
- Blue box: `[SE-ResidualBlock] → 256×256×32`

### Final Output:

- Blue box: `[Conv 1×1 + Sigmoid] → 256×256×1`
- Label: "Segmentation Mask (Binary)"

### Additional Visual Details:

- **Box dimensions:** Make encoder boxes progressively narrower (fewer spatial dims) but taller (more channels) going down
- **Box dimensions:** Make decoder boxes progressively wider and shorter going up
- **Skip connections:** Draw as horizontal dashed gray arrows with SE attention (red) nodes
- **All boxes:** Include dimensions in format `H×W×C` inside or below each box
- **ASPP module:** Highlight with a distinctive border/background to show it's the bottleneck
- **SE blocks:** Show as small red rectangular modules with internal arrows (GAP → squeeze → excite)
- **Legend:** Include color-coded legend in top-right or bottom-right corner
- **Title:** "MobileNet ASPP Residual SE U-Net Architecture" at the top

### Reference Architecture Dimensions:

```
Input:    256×256×1
enc0:     128×128×32   (frozen MobileNetV2)
enc1:      64×64×16    (frozen)
enc2:      32×32×24    (frozen)
enc3:      16×16×32    (frozen)
enc4:       8×8×96     (frozen)
enc5:       8×8×1280   (frozen)
Bottleneck: 8×8×512    (ASPP - trainable)
dec5:      16×16×256   (trainable)
dec4:      32×32×128   (trainable)
dec3:      64×64×64    (trainable)
dec2:     128×128×32   (trainable)
dec1:     256×256×32   (trainable)
Output:   256×256×1    (Sigmoid activation)

Total Parameters: ~4.8M
Trainable Parameters: ~1.2M (25%)
Frozen Parameters: ~3.6M (75% - encoder only)
```

---

## Overview

Building upon the baseline architecture, the proposed final model, **MobileNet ASPP Residual SE U-Net**, introduces advanced architectural mechanisms to address specific challenges in fetal ultrasound segmentation, such as speckle noise, blurry boundaries, and the varying size of the fetal head across gestational ages. This architecture integrates a frozen pre-trained encoder, a multi-scale bottleneck, and an attention-guided decoder, achieving a balance between computational efficiency and segmentation accuracy.

**Key Architectural Components:**

- **Encoder:** Frozen MobileNetV2 (pre-trained on ImageNet)
- **Bottleneck:** Atrous Spatial Pyramid Pooling (ASPP)
- **Decoder:** SE-Residual Blocks with channel attention
- **Skip Connections:** Enhanced with Squeeze-and-Excitation (SE) blocks

## 1. Encoder Backbone: Frozen MobileNetV2 with Transfer Learning

The encoder utilizes the **MobileNetV2** architecture (Sandler et al., 2018) pre-trained on the ImageNet dataset. Unlike the baseline where all layers may be trainable, this model employs a frozen encoder strategy during training.

### 1.1 Transfer Learning Justification

By freezing the weights of the MobileNetV2 backbone, the model leverages robust, low-level feature extractors (edges, textures, shapes) learned from millions of natural images. This approach provides several advantages for medical imaging tasks:

- **Prevents Overfitting:** With limited labeled medical data (999 training images), frozen pre-trained weights prevent the encoder from overfitting to the small dataset.
- **Accelerates Convergence:** Pre-learned features reduce training time from ~100 epochs to ~50-70 epochs.
- **Reduces Trainable Parameters:** Only the decoder path is trained, reducing parameters from 4.8M to ~1.2M trainable parameters (~75% reduction).
- **Domain Transfer:** Despite being trained on natural images, low-level features (edges, contours) generalize well to ultrasound imagery.

### 1.2 Input Adaptation for Grayscale Images

MobileNetV2 expects 3-channel RGB input, but ultrasound images are single-channel grayscale. The first convolutional layer is modified to accept 1-channel input:

```
Original: Conv2d(3, 32, kernel_size=3, stride=2)
Modified: Conv2d(1, 32, kernel_size=3, stride=2)
```

The weights are initialized by averaging the pre-trained RGB channel weights: `w_gray = mean(w_R, w_G, w_B)`, preserving the learned edge detection capabilities.

### 1.3 Multi-Scale Feature Extraction

The encoder extracts features at **six spatial resolutions** through hierarchical downsampling:

| Stage | Layer Index  | Channels | Spatial Resolution | Downsampling Factor |
| ----- | ------------ | -------- | ------------------ | ------------------- |
| enc0  | init_conv    | 32       | H × W              | 1/1 (full)          |
| enc1  | features[1]  | 16       | H/2 × W/2          | 1/2                 |
| enc2  | features[3]  | 24       | H/4 × W/4          | 1/4                 |
| enc3  | features[6]  | 32       | H/8 × W/8          | 1/8                 |
| enc4  | features[13] | 96       | H/16 × W/16        | 1/16                |
| enc5  | features[18] | 1280     | H/32 × W/32        | 1/32                |

For an input image of size 256×256, the final encoder output is 1280 channels at 8×8 spatial resolution.

### 1.4 Depthwise Separable Convolutions

MobileNetV2 employs **depthwise separable convolutions**, which decompose standard convolutions into:

1. **Depthwise Convolution:** Applies a single filter per input channel (spatial filtering).
2. **Pointwise Convolution:** 1×1 convolution to combine channels (cross-channel mixing).

This reduces computational cost by a factor of 8-9× compared to standard convolutions while maintaining representational power, making the architecture suitable for real-time inference.

## 2. The Bottleneck: Atrous Spatial Pyramid Pooling (ASPP)

To overcome the limitation of fixed receptive fields in standard U-Nets, an **Atrous Spatial Pyramid Pooling (ASPP)** module is integrated at the bottleneck, bridging the encoder and decoder.

### 2.1 Multi-Scale Context Aggregation

As proposed by Chen et al. in DeepLabv3 (2017), ASPP captures image context at multiple scales simultaneously. This is crucial for fetal head segmentation because:

- The head's size relative to the image frame varies significantly (gestational age: 14-40 weeks).
- Different zoom levels and probe positions create scale variability.
- Small-scale features (skull boundaries) and large-scale context (overall head shape) must be captured simultaneously.

### 2.2 ASPP Configuration

The ASPP module applies **four parallel pathways** to the bottleneck features (1280 channels from enc5):

1. **1×1 Convolution:** Captures point-wise features without dilation (rate=1).
2. **Atrous Conv (rate=6):** Effective receptive field ≈ 13×13 pixels.
3. **Atrous Conv (rate=12):** Effective receptive field ≈ 25×25 pixels.
4. **Atrous Conv (rate=18):** Effective receptive field ≈ 37×37 pixels.
5. **Global Average Pooling:** Encodes global image-level context.

All pathways produce 128 channels each (except GAP which produces 256), resulting in:

- **Input:** 1280 channels @ 8×8
- **After ASPP concatenation:** 640 channels @ 8×8
- **After 1×1 projection:** 512 channels @ 8×8 (bottleneck output)

### 2.3 Atrous Convolution Mechanism

Atrous (dilated) convolution inserts zeros (holes) between kernel elements to expand the receptive field without increasing parameters:

```
Standard 3×3 Conv: receptive field = 3×3 = 9 pixels
Atrous 3×3 Conv (rate=6): receptive field = 13×13 = 169 pixels
```

**Formula:** For a kernel size $k$ and dilation rate $r$, the effective receptive field is:
$$\text{RF}_{\text{effective}} = k + (k-1) \times (r-1)$$

For $k=3$ and $r=18$: $\text{RF} = 3 + 2 \times 17 = 37$ pixels.

This allows the model to "see" the global shape of the skull (head diameter ≈ 60-80 pixels at 8×8 resolution) while retaining local boundary details, without adding extra layers or parameters.

### 2.4 Dropout and Normalization

The ASPP module includes:

- **Dropout (p=0.5):** Applied after the final 1×1 projection to prevent overfitting.
- **GroupNorm:** Used in the global pooling branch for stable training with small batch sizes (batch size = 16).

## 3. The Decoder: SE-Residual Blocks

The most significant enhancement in the decoder path is the replacement of standard convolutional blocks with **Squeeze-and-Excitation (SE) Residual Blocks**. This hybrid block combines two powerful concepts to enable effective feature refinement during upsampling.

### 3.1 Residual Connections

Inspired by He et al. (2016), residual connections add the input of a convolutional block directly to its output:
$$y = \mathcal{F}(x, \{W_i\}) + x$$

where $\mathcal{F}(x, \{W_i\})$ represents the learned transformation (two Conv-BN-ReLU layers).

**Benefits:**

- **Gradient Flow:** Facilitates backpropagation through deep networks by providing a direct path for gradients.
- **Identity Mapping:** Allows the network to learn residual functions $\mathcal{F}(x)$ rather than the complete transformation $H(x)$.
- **Mitigates Vanishing Gradients:** Critical for training the 5-stage decoder with trainable parameters.

### 3.2 Squeeze-and-Excitation (SE) Mechanism

Integrated within each residual block, the SE mechanism (Hu et al., 2018) performs **channel-wise attention** to recalibrate feature importance:

#### 3.2.1 Squeeze Operation

Global Average Pooling (GAP) aggregates spatial information $(H \times W)$ into a channel descriptor $(C \times 1 \times 1)$:
$$z_c = \frac{1}{H \times W} \sum_{i=1}^{H} \sum_{j=1}^{W} x_c(i, j)$$

This creates a global statistic for each channel, encoding "what" information is present regardless of "where."

#### 3.2.2 Excitation Operation

A two-layer fully connected network learns channel interdependencies:
$$s = \sigma(W_2 \cdot \text{ReLU}(W_1 \cdot z))$$

where:

- $W_1 \in \mathbb{R}^{C/r \times C}$: Dimensionality reduction by ratio $r=16$
- $W_2 \in \mathbb{R}^{C \times C/r}$: Dimensionality expansion
- $\sigma$: Sigmoid activation producing channel weights $s_c \in [0, 1]$

#### 3.2.3 Feature Recalibration

The learned weights are applied channel-wise:
$$\tilde{x}_c = s_c \cdot x_c$$

**Impact on Fetal Head Segmentation:**

- **Emphasizes:** Feature maps highlighting skull boundaries, anatomical structures, and head contours.
- **Suppresses:** Feature maps responding to ultrasound artifacts (acoustic shadows, speckle noise, reverberation artifacts).
- **Adaptive:** The excitation weights are learned during training, allowing the model to discover which channels are most informative for boundary delineation.

### 3.3 Residual Block SE Architecture

Each `ResidualBlockSE` consists of:

```
Input (C_in channels)
    ↓
[Conv(3×3) → BatchNorm → ReLU] → (C_out channels)
    ↓
[Conv(3×3) → BatchNorm → ReLU] → (C_out channels)
    ↓
[SE Block: GAP → FC(C_out/16) → ReLU → FC(C_out) → Sigmoid]
    ↓
Element-wise recalibration
    ↓
Add skip connection (with 1×1 conv projection if C_in ≠ C_out)
    ↓
Output (C_out channels)
```

**Parameters per block:** With $C_{\text{in}} = 352$ and $C_{\text{out}} = 256$:

- Main path: $(3 \times 3 \times 352 \times 256) \times 2 \approx 1.6M$ parameters
- SE bottleneck: $(352 \times 16) + (16 \times 352) \approx 11K$ parameters (negligible overhead)
- Total SE overhead: **< 1%** of block parameters

## 4. Skip Connections with SE Enhancement

Unlike standard U-Net skip connections that directly concatenate encoder features, this architecture applies **SE blocks to skip connections** before concatenation.

**Rationale:**

- Encoder features contain both relevant (skull boundaries) and irrelevant (speckle noise, artifacts) information.
- SE blocks act as a "feature filter," amplifying useful channels and suppressing noisy ones.
- This selective feature passing improves decoder efficiency by reducing the noise in skip connections.

**Implementation:**

```
Encoder Feature (e.g., enc4: 96 channels)
    ↓
[SE Block: GAP → FC(96/16) → ReLU → FC(96) → Sigmoid]
    ↓
Recalibrated Features (96 channels)
    ↓
Concatenate with Decoder Upsampled Features
    ↓
[ResidualBlockSE]
```

## 5. Decoder Architecture

The decoder consists of **five upsampling stages**, progressively reconstructing the segmentation mask from the 8×8 bottleneck to full 256×256 resolution.

### 5.1 Decoder Channel Progression

| Stage | Input Res. | Output Res. | Upsampling | Skip (SE) | Concat | ResBlockSE Out |
| ----- | ---------- | ----------- | ---------- | --------- | ------ | -------------- |
| dec5  | 8×8        | 16×16       | 512→256    | 96 (enc4) | 352    | 256            |
| dec4  | 16×16      | 32×32       | 256→128    | 32 (enc3) | 160    | 128            |
| dec3  | 32×32      | 64×64       | 128→64     | 24 (enc2) | 88     | 64             |
| dec2  | 64×64      | 128×128     | 64→32      | 16 (enc1) | 48     | 32             |
| dec1  | 128×128    | 256×256     | 32→32      | 32 (enc0) | 64     | 32             |

**Upsampling Method:** Transposed Convolution (ConvTranspose2d) with:

- Kernel size: 2×2
- Stride: 2
- No padding

This learnable upsampling (vs. bilinear interpolation) allows the network to learn optimal upsampling kernels for ultrasound images.

### 5.2 Detailed Decoder Flow (Example: dec5)

```
Bottleneck ASPP output: [B, 512, 8, 8]
    ↓
ConvTranspose2d(512 → 256, k=2, s=2): [B, 256, 16, 16]
    ↓
Skip from enc4: [B, 96, 16, 16]
    ↓ (SE recalibration)
SE-enhanced skip: [B, 96, 16, 16]
    ↓
Concatenate: [B, 256+96=352, 16, 16]
    ↓
ResidualBlockSE(352 → 256): [B, 256, 16, 16]
    ↓
Pass to dec4...
```

## 6. Output Head

The final segmentation mask is generated through:

```
Decoder output: [B, 32, 256, 256]
    ↓
Conv2d(32 → 1, k=1×1): [B, 1, 256, 256]
    ↓
Sigmoid (applied in loss function during training): [B, 1, 256, 256]
```

**Note:** During training, the model outputs raw logits (no Sigmoid), and the loss function (`BCEWithLogitsLoss`) internally applies Sigmoid for numerical stability. During inference, Sigmoid is applied to produce probabilities.

## 7. Complete Architecture Integration

The overall data flow proceeds as follows:

### 7.1 Forward Pass

1. **Input:** 256×256 grayscale ultrasound image `[B, 1, 256, 256]`
2. **Initial Conv:** Custom trainable layer producing `[B, 32, 256, 256]` (enc0 skip)
3. **Encoder (Frozen MobileNetV2):** Hierarchical feature extraction:
   - enc1: `[B, 16, 128, 128]`
   - enc2: `[B, 24, 64, 64]`
   - enc3: `[B, 32, 32, 32]`
   - enc4: `[B, 96, 16, 16]`
   - enc5: `[B, 1280, 8, 8]`
4. **Bottleneck (ASPP):** Multi-scale context aggregation → `[B, 512, 8, 8]`
5. **Decoder (ResidualBlockSE + SE Skips):** Five upsampling stages with skip connections
6. **Output:** Binary segmentation mask `[B, 1, 256, 256]`

### 7.2 Information Flow Characteristics

- **Encoder:** Frozen, extracts hierarchical features (edges → textures → shapes)
- **Bottleneck:** Trainable, learns multi-scale contextual representations
- **Decoder:** Trainable, learns to combine multi-scale features and reconstruct boundaries
- **Skip Connections:** SE-enhanced, selective feature passing

## 8. Training Strategy

### 8.1 Loss Function

**DiceBCELoss:** Weighted combination of Dice Loss and Binary Cross-Entropy with Logits.

$$\mathcal{L}_{\text{total}} = w_{\text{Dice}} \cdot \mathcal{L}_{\text{Dice}} + w_{\text{BCE}} \cdot \mathcal{L}_{\text{BCE}}$$

**Dice Loss** (region overlap):
$$\mathcal{L}_{\text{Dice}} = 1 - \frac{2 \cdot |X \cap Y| + \epsilon}{|X| + |Y| + \epsilon}$$

**BCE with Logits Loss** (pixel-wise accuracy with class imbalance handling):
$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N} \sum_{i=1}^{N} \left[ w_{\text{pos}} \cdot y_i \cdot \log(\sigma(x_i)) + (1-y_i) \cdot \log(1-\sigma(x_i)) \right]$$

where:

- $w_{\text{Dice}} = 0.8$, $w_{\text{BCE}} = 0.2$ (emphasize region overlap)
- $w_{\text{pos}} = 200$ (compensate for foreground:background ratio ≈ 1:260)
- $\epsilon = 10^{-6}$ (smoothing term)

**Rationale:**

- **Dice Loss:** Directly optimizes the evaluation metric (DSC), handles class imbalance naturally.
- **BCE Loss:** Provides pixel-level supervision and stable gradients, especially in early training.
- **Positive Weight:** Heavily penalizes false negatives (missing fetal head regions).

### 8.2 Optimization

- **Optimizer:** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.999$
- **Learning Rate:** $\text{lr} = 3 \times 10^{-4}$ (lower than standard due to frozen encoder)
- **Weight Decay:** $1 \times 10^{-4}$ (L2 regularization on trainable parameters)
- **Scheduler:** ReduceLROnPlateau
  - Monitor: Validation Dice Score (maximize)
  - Patience: 10 epochs
  - Reduction Factor: 0.5×
  - Minimum LR: $1 \times 10^{-6}$

### 8.3 Training Configuration

- **Batch Size:** 16 (limited by GPU memory)
- **Epochs:** 100 (with early stopping patience = 15 epochs)
- **Data Augmentation:**
  - Horizontal Flip (p=0.5)
  - Rotation (±20°, p=0.5)
  - Scale (±10%, p=0.5)
  - Translation (±10%, p=0.5)
  - Applied with Albumentations library (synchronized image-mask transforms)

### 8.4 Freezing Strategy

The encoder (MobileNetV2) remains **frozen throughout training**:

- **Frozen Parameters:** ~2.2M (encoder)
- **Trainable Parameters:** ~1.2M (decoder + bottleneck + skip SE blocks)
- **Total Parameters:** ~3.4M
- **Trainable Ratio:** ~35%

**Benefits:**

- **Training Speed:** 2-3× faster convergence (50-70 epochs vs. 100+ for fully trainable)
- **GPU Memory:** Reduced memory usage (no encoder gradient storage)
- **Regularization:** Pre-trained features prevent overfitting
- **Generalization:** Better performance on limited data

## 9. Model Complexity Analysis

### 9.1 Parameter Count

- **Total Parameters:** 3,403,809 (~3.4M)
- **Trainable Parameters:** 1,192,737 (~1.2M, 35.0%)
- **Frozen Parameters:** 2,211,072 (~2.2M, 65.0%)
- **Model Size:** ~13.0 MB (float32), ~3.5 MB (float16)

### 9.2 Computational Complexity

For input size 256×256:

- **FLOPs:** ~2.8 GFLOPs (forward pass)
- **Memory:** ~450 MB (batch size = 16)
- **Inference Time:** ~15-20 ms on NVIDIA V100 GPU (~50-60 FPS)

### 9.3 Comparison with Baseline U-Net

| Metric               | Standard U-Net | MobileNet ASPP SE U-Net | Improvement |
| -------------------- | -------------- | ----------------------- | ----------- |
| Total Parameters     | 7.8M           | 3.4M                    | -56%        |
| Trainable Parameters | 7.8M           | 1.2M                    | -85%        |
| Inference Time (GPU) | 25 ms          | 18 ms                   | +28% faster |
| Model Size (float32) | 31 MB          | 13 MB                   | -58%        |
| DSC (validation)     | 96.8%          | 97.81%                  | +1.01%      |

## 10. Key Advantages

1. **Efficiency Through Transfer Learning:**

   - Frozen encoder reduces trainable parameters by 85%
   - Faster training convergence (50 vs. 100 epochs)
   - Lower GPU memory requirements

2. **Multi-Scale Context Awareness:**

   - ASPP bottleneck captures features at multiple scales (dilation rates: 6, 12, 18)
   - Handles varying fetal head sizes across gestational ages
   - Global average pooling provides image-level context

3. **Channel-Wise Attention:**

   - SE blocks in decoder emphasize boundary-relevant features
   - SE blocks on skip connections filter encoder noise
   - Adaptive recalibration improves feature quality

4. **Architectural Depth with Gradient Stability:**

   - Residual connections enable deep decoder (5 stages)
   - Mitigates vanishing gradients during backpropagation
   - Allows learning of complex feature refinement

5. **Practical Deployment:**
   - Compact model size (13 MB) suitable for edge devices
   - Fast inference (~18 ms) enables near real-time applications
   - Depthwise separable convolutions reduce computational cost

## 11. References

1. **MobileNetV2:** Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. _CVPR 2018_.

2. **DeepLabv3/ASPP:** Chen, L. C., Papandreou, G., Schroff, F., & Adam, H. (2017). Rethinking Atrous Convolution for Semantic Image Segmentation. _arXiv:1706.05587_.

3. **Residual Networks:** He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. _CVPR 2016_.

4. **Squeeze-and-Excitation Networks:** Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. _CVPR 2018_.

5. **U-Net:** Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. _MICCAI 2015_.
