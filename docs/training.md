# Training Guide

## Quick Start

```bash
python scripts/train.py --config configs/models/mobinet_aspp_residual_se.yaml
```

## Configuration

Training parameters are specified in YAML config files in `configs/models/`.

### Key Configuration Parameters

```yaml
# Model Configuration
model:
  name: 'mobinet_aspp_residual_se'
  input_channels: 1
  output_channels: 1

# Training Configuration
training:
  batch_size: 16
  epochs: 100
  learning_rate: 0.001
  optimizer: 'adam'

# Data Configuration
data:
  train_path: 'data/processed/training_set'
  val_path: 'data/processed/validation_set'
  test_path: 'data/processed/test_set'
  image_size: [256, 256]

# Augmentation
augmentation:
  horizontal_flip: 0.5
  rotation_limit: 20
  scale_limit: 0.1
  translate_limit: 0.1
```

## Training Process

### 1. Data Loading

- Images are loaded from `data/processed/`
- Preprocessing: grayscale, resize to 256×256, normalize [0,1]
- Augmentation applied during training

### 2. Model Initialization

- Model architecture loaded from config
- Weights initialized randomly or from checkpoint

### 3. Training Loop

```python
for epoch in range(num_epochs):
    # Training phase
    for batch in train_loader:
        outputs = model(images)
        loss = criterion(outputs, masks)
        optimizer.step()

    # Validation phase
    for batch in val_loader:
        outputs = model(images)
        val_loss = criterion(outputs, masks)
        metrics = calculate_metrics(outputs, masks)

    # Learning rate scheduling
    scheduler.step(val_loss)

    # Checkpoint saving
    if val_loss < best_loss:
        save_checkpoint(model, optimizer, epoch)
```

### 4. Loss Function

Combined loss: `Dice Loss + Binary Cross-Entropy`

### 5. Optimizer

Adam optimizer with initial learning rate 0.001

### 6. Learning Rate Scheduling

ReduceLROnPlateau:

- Monitors validation loss
- Reduces LR when plateau detected
- Factor: 0.5
- Patience: 10 epochs

## Monitoring Training

### Checkpoints

- Saved in `models/checkpoints/`
- Best model saved in `models/best/`
- Includes: model weights, optimizer state, epoch number

### Logs

- Training logs saved in `experiments/logs/`
- Includes: loss curves, learning rate, metrics

### Visualizations

- Generated in `outputs/visualizations/`
- Includes: prediction examples, loss curves

## Resume Training

```bash
python scripts/train.py \
    --config configs/models/mobinet_aspp_residual_se.yaml \
    --resume models/checkpoints/checkpoint_epoch_50.pth
```

## Multi-GPU Training

```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/train.py \
    --config configs/models/mobinet_aspp_residual_se.yaml \
    --multi-gpu
```

## Hyperparameter Tuning

Key hyperparameters to tune:

1. **Learning Rate** (0.0001 - 0.01)
2. **Batch Size** (8, 16, 32)
3. **Augmentation Probabilities** (0.3 - 0.7)
4. **Model Capacity** (number of filters)

## Expected Results

Target metrics on test set:

- **DSC:** ≥97.81%
- **mIoU:** ≥97.90%
- **mPA:** ≥99.18%

Training typically converges in 80-100 epochs.

## Common Issues

### Overfitting

- Increase augmentation strength
- Add dropout layers
- Reduce model capacity

### Underfitting

- Increase model capacity
- Train for more epochs
- Reduce regularization

### Training Instability

- Reduce learning rate
- Use gradient clipping
- Check for data issues

## Evaluation

After training:

```bash
python scripts/evaluate.py \
    --model models/best/best_model.pth \
    --config configs/models/mobinet_aspp_residual_se.yaml
```

See evaluation results in `outputs/reports/`.
