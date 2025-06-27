#!/bin/bash

echo "🔧 FaceID Dependencies Installation Script"
echo "==========================================="

# Update pip first
echo "📦 Updating pip..."
pip install --upgrade pip

# Install packages one by one for better error tracking
echo "🚀 Installing core packages..."

# Essential packages first
pip install streamlit
pip install numpy
pip install pillow
pip install opencv-python
pip install matplotlib
pip install pandas
pip install scikit-learn

# Face detection
echo "👁️ Installing face detection..."
pip install mtcnn

# TensorFlow (try different versions)
echo "🧠 Installing TensorFlow..."
if pip install tensorflow; then
    echo "✅ TensorFlow installed successfully"
elif pip install tensorflow-cpu; then
    echo "✅ TensorFlow CPU installed successfully"
elif pip install "tensorflow==2.12.0"; then
    echo "✅ TensorFlow 2.12.0 installed successfully"
else
    echo "❌ TensorFlow installation failed"
    echo "Please install TensorFlow manually:"
    echo "  pip install tensorflow"
fi

echo ""
echo "🎉 Installation completed!"
echo "Run: streamlit run app.py" 