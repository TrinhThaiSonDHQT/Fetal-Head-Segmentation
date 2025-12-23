"""
Fetal Head Segmentation in Ultrasound Images using an Improved U-Net
Setup configuration for package installation
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fetal-head-segmentation",
    version="1.0.0",
    author="Trinh Thai Son",
    description="Deep learning model for fetal head segmentation in ultrasound images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TrinhThaiSonDHQT/Fetal-Head-Segmentation",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "opencv-python>=4.5.0",
        "albumentations>=1.0.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "scikit-learn>=0.24.0",
        "Pillow>=8.0.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.60.0",
        "pandas>=1.2.0",
        "scipy>=1.6.0",
    ],
    extras_require={
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "black>=21.0",
            "flake8>=3.9.0",
            "pytest>=6.2.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "fhs-train=scripts.train:main",
            "fhs-evaluate=scripts.evaluate:main",
            "fhs-predict=scripts.predict:main",
        ],
    },
)
