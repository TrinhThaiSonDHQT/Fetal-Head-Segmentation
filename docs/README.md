# Fetal Head Segmentation

High-accuracy deep learning model for fetal head segmentation in 2D ultrasound images using an improved U-Net architecture.

## Project Structure

```
fetal-head-segmentation/
├── configs/              # Configuration files (YAML)
├── data/                 # Dataset storage
│   ├── raw/             # Original HC18 dataset
│   └── processed/       # Preprocessed train/val/test splits
├── docs/                # Documentation
├── experiments/         # Experiment logs and tracking
├── models/              # Saved model weights
│   ├── best/           # Best performing models
│   └── checkpoints/    # Training checkpoints
├── notebooks/           # Jupyter notebooks for analysis
├── outputs/             # Generated outputs
│   ├── predictions/    # Model predictions
│   ├── visualizations/ # Result visualizations
│   └── reports/        # Evaluation reports
├── scripts/             # Entry point scripts
│   ├── train.py        # Training script
│   ├── evaluate.py     # Evaluation script
│   ├── predict.py      # Inference script
│   └── prepare_dataset.py  # Data preprocessing
└── src/                 # Source code package
    ├── data/           # Data loading and preprocessing
    ├── models/         # Model architectures
    ├── losses/         # Loss functions
    ├── metrics/        # Evaluation metrics
    ├── training/       # Training logic
    ├── inference/      # Inference logic
    └── utils/          # Utility functions
```

## Installation

```bash
# Clone repository
git clone https://github.com/TrinhThaiSonDHQT/Fetal-Head-Segmentation.git
cd Fetal-Head-Segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .
```

## Quick Start

### Training

```bash
python scripts/train.py --config configs/models/mobinet_aspp_residual_se.yaml
```

### Evaluation

```bash
python scripts/evaluate.py --model models/best/best_model.pth --config configs/models/mobinet_aspp_residual_se.yaml
```

### Inference

```bash
python scripts/predict.py --model models/best/best_model.pth --input path/to/image.png --output path/to/output/
```

## Model Architecture

Improved U-Net with:

- **Residual Blocks** in encoder/decoder
- **Feature Pyramid + Scale Attention Module (FP+SAM)** for multi-scale feature fusion
- **ASPP (Atrous Spatial Pyramid Pooling)** for enhanced contextual information

## Performance Metrics

- **DSC (Dice Similarity Coefficient):** ≥97.81%
- **mIoU (Mean Intersection over Union):** ≥97.90%
- **mPA (Mean Pixel Accuracy):** ≥99.18%

## Dataset

HC18 Grand Challenge dataset:

- Training: 999 images
- Testing: 355 images
- Resolution: 256×256 pixels

## License

MIT License - see LICENSE file for details

## Contact

**Trinh Thai Son**  
IT Student, Final Year Thesis Project  
GitHub: [@TrinhThaiSonDHQT](https://github.com/TrinhThaiSonDHQT)
