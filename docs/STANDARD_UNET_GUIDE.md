# Standard U-Net Implementation Guide

## Overview

This guide provides instructions for using the **Standard U-Net** baseline model for fetal head segmentation.

**Note:** All components work seamlessly on both **CPU and GPU**. See [CPU/GPU Compatibility Guide](CPU_GPU_COMPATIBILITY.md) for details.

## Key Features

The Standard U-Net implementation includes:

1. **Standard U-Net Architecture**

   - Encoder-decoder with skip connections
   - 4 downsampling blocks (64 → 128 → 256 → 512 channels)
   - Bottleneck: 1024 channels
   - 4 upsampling blocks (512 → 256 → 128 → 64 channels)

2. **ReLU Activations**

   - All convolutional layers use ReLU activation

3. **Padded Convolutions**

   - `padding=1` (equivalent to `padding='same'`) maintains spatial dimensions

4. **Image Normalization**

   - Input images normalized by dividing by 255.0

5. **Training Configuration**
   - Optimizer: Adam with learning rate 1e-4
   - Loss: Binary Cross-Entropy (BCELoss)
   - Batch size: 16
   - Epochs: 100 with early stopping

## Quick Start

### Option 1: Using the Training Script

```bash
# From project root
python train_standard_unet.py
```

### Option 2: Using the Jupyter Notebook

1. Open the notebook:

   ```
   notebooks/accuracy_focus/standard_u_net/Standard_U-Net.ipynb
   ```

2. Run all cells sequentially

3. The notebook provides:
   - Model initialization
   - Data loading with normalization
   - Training loop
   - Visualization of results

## File Structure

```
├── src/
│   ├── models/
│   │   └── standard_unet.py          # Standard U-Net model
│   ├── metrics/
│   │   └── segmentation_metrics.py   # Dice, IoU, Pixel Accuracy
│   └── utils/
│       └── visualization.py          # Visualization utilities
├── configs/
│   └── standard_unet_config.yaml     # Configuration file
├── train_standard_unet.py             # Training script
└── notebooks/
    └── accuracy_focus/
        └── standard_u_net/
            └── Standard_U-Net.ipynb   # Training notebook
```

## Configuration

Edit `configs/standard_unet_config.yaml` to customize:

```yaml
model:
  name: 'StandardUNet'
  in_channels: 1
  out_channels: 1
  base_filters: 64

training:
  batch_size: 16
  num_epochs: 100
  optimizer:
    name: 'Adam'
    lr: 0.0001 # 1e-4

loss:
  name: 'BCELoss' # Binary Cross-Entropy
```

## Model Architecture Details

### Encoder (Contracting Path)

| Block | Input Channels | Output Channels | Operations           |
| ----- | -------------- | --------------- | -------------------- |
| inc   | 1              | 64              | DoubleConv           |
| down1 | 64             | 128             | MaxPool → DoubleConv |
| down2 | 128            | 256             | MaxPool → DoubleConv |
| down3 | 256            | 512             | MaxPool → DoubleConv |
| down4 | 512            | 1024            | MaxPool → DoubleConv |

### Decoder (Expansive Path)

| Block | Input Channels | Skip Channels | Output Channels | Operations                          |
| ----- | -------------- | ------------- | --------------- | ----------------------------------- |
| up1   | 1024           | 512           | 512             | ConvTranspose → Concat → DoubleConv |
| up2   | 512            | 256           | 256             | ConvTranspose → Concat → DoubleConv |
| up3   | 256            | 128           | 128             | ConvTranspose → Concat → DoubleConv |
| up4   | 128            | 64            | 64              | ConvTranspose → Concat → DoubleConv |

### DoubleConv Block

```
Conv2d(kernel=3, padding=1) → BatchNorm2d → ReLU →
Conv2d(kernel=3, padding=1) → BatchNorm2d → ReLU
```

## Preprocessing Pipeline

1. **Load Image**: Grayscale ultrasound image
2. **Resize**: To 256×256 pixels
3. **Normalize**: Divide by 255.0 (range [0, 1])
4. **Convert**: To PyTorch tensor

## Training Pipeline

1. **Forward Pass**: Input → Model → Output (with sigmoid)
2. **Loss Calculation**: Binary Cross-Entropy between output and ground truth
3. **Backward Pass**: Compute gradients
4. **Optimizer Step**: Update weights using Adam (lr=1e-4)
5. **Validation**: Calculate Dice, IoU, Pixel Accuracy
6. **Learning Rate Scheduling**: ReduceLROnPlateau based on validation Dice

## Evaluation Metrics

- **Dice Coefficient (DSC)**: Measures overlap between prediction and ground truth
- **IoU Score (Jaccard Index)**: Intersection over union
- **Pixel Accuracy (PA)**: Percentage of correctly classified pixels

## Expected Results

After training, you should see:

- Training/validation loss curves
- Dice coefficient progression
- Sample predictions with metrics
- Best model checkpoint saved at `results/checkpoints/standard_unet/best_model.pth`

## Comparison with Improved U-Net

| Feature               | Standard U-Net | Improved U-Net |
| --------------------- | -------------- | -------------- |
| Basic Architecture    | ✓              | ✓              |
| Skip Connections      | ✓              | ✓              |
| Residual Blocks       | ✗              | ✓              |
| ASPP Module           | ✗              | ✓              |
| Feature Pyramid + SAM | ✗              | ✓              |
| Parameters            | ~31M           | ~35M           |
| Loss Function         | BCE            | Dice + BCE     |

## Troubleshooting

### Issue: Out of Memory (OOM)

**Solution**: Reduce batch size in config:

```yaml
training:
  batch_size: 8 # or lower
```

### Issue: Poor Performance

**Checklist**:

1. Verify data normalization (dividing by 255.0)
2. Check learning rate (should be 1e-4)
3. Ensure proper data augmentation
4. Verify loss function (BCELoss)

### Issue: Model Not Learning

**Checklist**:

1. Check loss values (should decrease)
2. Verify gradient flow (no NaN values)
3. Inspect data (correct image-mask pairs)
4. Check learning rate (not too low or high)

## Next Steps

1. **Train the Model**: Run training script or notebook
2. **Monitor Progress**: Watch loss and Dice coefficient
3. **Evaluate Results**: Compare with improved U-Net baseline
4. **Experiment**: Try different hyperparameters if needed

## References

- Original U-Net Paper: [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597)
- HC18 Grand Challenge: [Medical Image Analysis](https://hc18.grand-challenge.org/)

## Contact

For issues or questions about this implementation, refer to the project documentation or create an issue in the repository.
