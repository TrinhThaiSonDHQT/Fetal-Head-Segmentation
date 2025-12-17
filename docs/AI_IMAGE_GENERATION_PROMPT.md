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
