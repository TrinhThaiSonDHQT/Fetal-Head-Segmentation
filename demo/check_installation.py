#!/usr/bin/env python3
"""
Installation Validator for Demo Platform
Checks all dependencies and requirements before running
"""
import sys
import subprocess
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("❌ Python 3.7+ required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not installed")
        return False


def check_model_file():
    """Check if model file exists"""
    model_path = Path('best_models/best_model_mobinet_aspp_residual_se_v2.pth')
    
    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 ** 2)
        print(f"✅ Model file found ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ Model file not found: {model_path}")
        print("   Please ensure the model is in the correct location")
        return False


def check_gpu():
    """Check if GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name}")
            return True
        else:
            print("⚠️  GPU not available (will use CPU)")
            return True
    except:
        print("⚠️  Cannot check GPU status")
        return True


def install_requirements():
    """Install requirements from requirements.txt"""
    print("\n📦 Installing dependencies...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All dependencies installed")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install dependencies")
        return False


def main():
    print("="*60)
    print("Demo Platform - Installation Validator")
    print("="*60)
    print()
    
    # Change to demo directory
    demo_dir = Path(__file__).parent
    import os
    os.chdir(demo_dir)
    
    all_ok = True
    
    # Check Python version
    print("🔍 Checking Python version...")
    if not check_python_version():
        all_ok = False
    
    print("\n🔍 Checking required packages...")
    packages = [
        ('torch', 'torch'),
        ('torchvision', 'torchvision'),
        ('gradio', 'gradio'),
        ('opencv-python', 'cv2'),
        ('Pillow', 'PIL'),
        ('numpy', 'numpy'),
    ]
    
    missing_packages = []
    for pkg_name, import_name in packages:
        if not check_package(pkg_name, import_name):
            missing_packages.append(pkg_name)
            all_ok = False
    
    # Offer to install missing packages
    if missing_packages:
        print(f"\n⚠️  Found {len(missing_packages)} missing package(s)")
        response = input("Install missing packages now? (y/n): ").lower()
        if response == 'y':
            if install_requirements():
                all_ok = True
                print("\n✅ All packages installed successfully")
            else:
                all_ok = False
    
    # Check model file
    print("\n🔍 Checking model file...")
    if not check_model_file():
        all_ok = False
    
    # Check GPU
    print("\n🔍 Checking GPU availability...")
    check_gpu()
    
    # Final status
    print("\n" + "="*60)
    if all_ok:
        print("✅ All checks passed! Ready to launch demo.")
        print("\nTo start the demo, run:")
        print("  python app.py")
    else:
        print("❌ Some checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing packages: pip install -r requirements.txt")
        print("  - Verify model file location")
    print("="*60)
    
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
