# Dataset V5: Large-Scale Fetal Head Biometry Dataset

## Overview
This dataset contains fetal head ultrasound images with multi-class segmentation masks.

## Dataset Split
- **Training Set**: 70% of patients
- **Validation Set**: 15% of patients  
- **Test Set**: 15% of patients

## Split Strategy ✅ NO DATA LEAKAGE
- **TRUE patient-level split**: Each patient is assigned to EXACTLY ONE split
- **All images from a patient go to the same split**: Prevents information leakage
- **Global patient splitting**: Patients are split first, then all their images (across all planes) are assigned together
- **Natural plane distribution**: Maintains anatomical plane ratios without artificial stratification

## Key Improvements Over V4
- **V4 Issue**: Patients were split per-plane, causing same patient to appear in multiple splits
- **V5 Solution**: Patients split globally across all planes, ensuring zero leakage
- **Verified**: No patient appears in more than one split

## Anatomical Planes
1. **Trans-cerebellum**: Cerebellum plane images
2. **Trans-thalamic**: Thalamic plane images
3. **Trans-ventricular**: Ventricular plane images
4. **Diverse**: Mixed fetal head images

## Segmentation Classes
Based on PASCAL VOC format with the following labels:
- **Background** (RGB: 0,0,0)
- **Brain** (RGB: 128,0,0)
- **CSP** - Cavum Septum Pellucidum (RGB: 0,128,0)
- **LV** - Lateral Ventricles (RGB: 128,128,0)

## Directory Structure
```
dataset_v5/
├── training_set/
│   ├── images/
│   └── masks/
├── validation_set/
│   ├── images/
│   └── masks/
├── test_set/
│   ├── images/
│   └── masks/
├── dataset_metadata.csv
└── README.md
```

## Metadata File
`dataset_metadata.csv` contains:
- `filename`: Image filename
- `split`: Dataset split (training_set/validation_set/test_set)
- `plane`: Anatomical plane category
- `patient_id`: Patient identifier (verify no patient in multiple splits!)
- `pixel_size_mm`: Real-world pixel size in millimeters

## Source
Large-Scale Annotation Dataset for Fetal Head Biometry

## Notes
- Images are in PNG format
- Masks use PASCAL VOC color encoding
- Pixel size calibration data included for accurate biometry measurements
- Random seed: 42 (for reproducibility)
- **IMPORTANT**: Verify no data leakage by checking patient_id distribution across splits
