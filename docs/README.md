# Fetal Head Segmentation

High-accuracy deep learning model for fetal head segmentation in 2D ultrasound images using an improved U-Net architecture.

## Project Structure

```
fetal-head-segmentation/
├── configs/              # Configuration files (YAML)
│   ├── mobinet_aspp_residual_se_config.yaml
│   └── standard_mobinet_unet_config.yaml
├── data/                 # Dataset storage
│   ├── raw/             # Original dataset
│   └── processed/       # Preprocessed train/val/test splits
│       ├── training_set/
│       ├── validation_set/
│       ├── test_set/
│       └── dataset_metadata.csv
├── docs/                # Documentation
├── efficient_focus/     # Efficient model implementations
│   ├── src/
│   ├── configs/
│   ├── results/
│   └── weights/
├── models/              # Saved model weights
│   └── best/           # Best performing models
├── notebooks/           # Jupyter notebooks for analysis
│   ├── mobinet_aspp_residual_se/
│   └── standard_mobinet_unet/
├── outputs/             # Generated outputs
│   ├── checkpoints/    # Training checkpoints
│   ├── logs/           # Training logs
│   ├── predictions/    # Model predictions
│   └── visualizations/ # Result visualizations
├── scripts/             # Utility scripts
│   └── prepare_dataset.py  # Data preprocessing
├── shared/              # Shared utilities and resources
└── src/                 # Source code package
    ├── data/           # Data loading and preprocessing
    │   ├── dataset.py
    │   └── dataset_v2.py
    ├── losses/         # Loss functions
    │   ├── bce_logits.py
    │   └── dice_bce_loss.py
    ├── metrics/        # Evaluation metrics
    │   ├── dice_score.py
    │   ├── iou.py
    │   ├── pixel_accuracy.py
    │   └── segmentation_metrics.py
    ├── models/         # Model architectures
    │   ├── components/
    │   └── variants/
    └── utils/          # Utility functions
        ├── device_utils.py
        ├── logger.py
        ├── optimizer.py
        ├── saver.py
        ├── train.py
        ├── visualization.py
        └── transforms/
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

### Exploratory Data Analysis

```bash
# Run notebooks for data exploration and visualization
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
jupyter notebook notebooks/02_visualize_augmentation_process.ipynb
```

### Training & Evaluation

Refer to the notebooks in `notebooks/mobinet_aspp_residual_se/` or `notebooks/standard_mobinet_unet/` for complete training and evaluation workflows.

### Using Pretrained Models

Pretrained models are available in `models/best/`:

- `best_model_mobinet_aspp_residual_se_v2.pth`
- `best_model_mobinet_aspp_residual_se_v3.pth`

## Model Architecture

Improved U-Net with:

- **Residual Blocks** in encoder/decoder
- **Feature Pyramid + Scale Attention Module (FP+SAM)** for multi-scale feature fusion
- **ASPP (Atrous Spatial Pyramid Pooling)** for enhanced contextual information

## Dataset

**Large-Scale Annotation Dataset for Fetal Head Biometry**

- **Total Images:** 3,792 ultrasound images
- **Original Dimensions:** 959 × 661 pixels (preprocessed to 256×256)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Fetal Plane Groups:**
  - Trans-thalamic
  - Trans-ventricular
  - Trans-cerebellum
  - Diverse fetal head images
- **Class Instances:**
  - Brain: 3,794 instances
  - CSP (Cavum Septum Pellucidum): 1,865 instances
  - LV (Lateral Ventricle): 1,512 instances
- **Available Formats:** 11 formats including COCO, YOLO, PASCAL, Segmentation mask, and others
- **Data Split:** Custom train/validation/test splits created during preprocessing

For more details, see [docs/Readme.txt](Readme.txt).

## License

MIT License - see LICENSE file for details

## Contact

**Trinh Thai Son**  
IT Student, Final Year Thesis Project  
GitHub: [@TrinhThaiSonDHQT](https://github.com/TrinhThaiSonDHQT)
