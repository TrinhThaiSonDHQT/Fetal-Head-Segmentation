# Setup Guide

## Prerequisites

- Python 3.7 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 50GB+ disk space (for dataset and outputs)

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/TrinhThaiSonDHQT/Fetal-Head-Segmentation.git
cd Fetal-Head-Segmentation
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

## Dataset Setup

### Option 1: Use Preprocessed Data

If you have already preprocessed data in `data/processed/`:

- `data/processed/training_set/`
- `data/processed/validation_set/`
- `data/processed/test_set/`

Skip to training.

### Option 2: Prepare from Raw Data

```bash
# Ensure raw data is in data/raw/
# Run preprocessing script
python scripts/prepare_dataset.py
```

## Configuration

Configuration files are in `configs/`:

- `mobinet_aspp_residual_se_config.yaml` - Full model configuration
- `standard_mobinet_unet_config.yaml` - Baseline model configuration

Edit these files to customize:

- Training hyperparameters
- Data paths
- Model architecture parameters
- Augmentation settings

## Directory Structure Setup

Ensure the following directories exist (they should be created automatically):

```bash
mkdir -p models/best
mkdir -p outputs/checkpoints outputs/logs outputs/predictions outputs/visualizations
mkdir -p data/raw data/processed
```

## Common Issues

### CUDA Out of Memory

- Reduce batch size in config file
- Use smaller image resolution (though 256×256 is recommended)

### Import Errors

- Ensure you're in the project root directory
- Verify virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

### Dataset Not Found

- Check paths in config files
- Verify dataset structure matches expected format
- Run `prepare_dataset.py` if preprocessing is needed

## Next Steps

After setup:

1. Review [architecture.md](architecture.md) for model details
2. Explore notebooks in `notebooks/` for:
   - Data analysis and visualization
   - Model training and evaluation
   - Statistical validation of results
