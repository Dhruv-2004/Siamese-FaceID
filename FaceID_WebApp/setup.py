#!/usr/bin/env python3
"""
Setup script for FaceID Web Application
This script helps with installation and initial setup.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python version: {platform.python_version()}")
    return True

def install_requirements():
    """Install required packages with fallback options"""
    
    # Check if we're in a conda environment
    is_conda = 'CONDA_DEFAULT_ENV' in os.environ or 'CONDA_PREFIX' in os.environ
    
    if is_conda:
        print("🐍 Conda environment detected. Using conda-compatible installation...")
        
        # Try conda first
        try:
            print("📦 Installing packages via conda...")
            subprocess.check_call(["conda", "install", "-y", "-c", "conda-forge", 
                                 "streamlit", "tensorflow", "opencv", "numpy", 
                                 "pillow", "matplotlib", "pandas", "scikit-learn"])
            
            # Install mtcnn via pip since it's not available in conda-forge
            print("📦 Installing mtcnn via pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "mtcnn"])
            
            print("✅ Requirements installed successfully via conda/pip")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Conda installation failed, falling back to pip...")
    
    # Try the main requirements file
    try:
        print("📦 Installing requirements via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Main requirements failed, trying minimal requirements...")
        
        # Try minimal requirements
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements-minimal.txt"])
            print("✅ Minimal requirements installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("⚠️  Minimal requirements failed, trying individual packages...")
            
            # Try installing packages individually
            essential_packages = [
                "streamlit",
                "opencv-python",
                "numpy",
                "pillow",
                "matplotlib",
                "pandas",
                "mtcnn"
            ]
            
            failed_packages = []
            for package in essential_packages:
                try:
                    print(f"📦 Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                except subprocess.CalledProcessError:
                    failed_packages.append(package)
            
            # Try TensorFlow last (often the most problematic)
            try:
                print("📦 Installing TensorFlow...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
            except subprocess.CalledProcessError:
                failed_packages.append("tensorflow")
                print("⚠️  TensorFlow installation failed. Trying CPU-only version...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-cpu"])
                except subprocess.CalledProcessError:
                    print("❌ All TensorFlow installations failed")
            
            if failed_packages:
                print(f"⚠️  Failed to install: {', '.join(failed_packages)}")
                print("You may need to install these manually.")
                return len(failed_packages) < len(essential_packages) + 1  # Allow some failures
            else:
                print("✅ All packages installed successfully")
                return True

def create_directories():
    """Create necessary directories"""
    directories = ["user_data", "static", "templates"]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"📁 Created directory: {directory}")
        else:
            print(f"✅ Directory exists: {directory}")

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "../models/siamese_network.h5",
        "../models/training_checkpoints/siamese_ckpt.weights.h5"
    ]
    
    found_model = False
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"✅ Found model file: {model_file}")
            found_model = True
            break
    
    if not found_model:
        print("⚠️  No model files found. Please ensure you have:")
        print("   - ../models/siamese_network.h5 OR")
        print("   - ../models/training_checkpoints/siamese_ckpt.weights.h5")
        print("   Copy these from your original FaceID project.")
    
    return found_model

def main():
    """Main setup function"""
    print("🚀 FaceID Web Application Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check model files
    model_exists = check_model_files()
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    
    if model_exists:
        print("\n🚀 You can now run the application with:")
        print("   streamlit run app.py")
    else:
        print("\n⚠️  Please copy your trained model files before running the app.")
    
    print("\n📖 For more information, see README.md")

if __name__ == "__main__":
    main() 