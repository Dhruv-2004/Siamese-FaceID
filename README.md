# FaceID - Siamese Neural Network Face Verification System

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/tensorflow-v2.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/streamlit-v1.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A sophisticated face verification system built with Siamese Neural Networks, implementing the groundbreaking "One-shot Image Recognition" research by Koch et al. (2015). This project demonstrates an innovative application of their Siamese network architecture for real-world face verification, featuring real-time face detection, secure user registration, and video-based face extraction for local deployment.

## 🧠 Research Foundation

This project is built upon the seminal work **"Siamese Neural Networks for One-shot Image Recognition"** by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov. Their research introduced a novel approach to learning similarity metrics through Siamese networks, enabling accurate classification with minimal training examples.

### 🚀 Innovation: From Research to Real-World Application

While the original paper focused on general one-shot image recognition, this project **innovatively adapts their methodology specifically for face verification**:

- **Original Research Focus**: Generic one-shot learning for various image classification tasks
- **Our Innovation**: Specialized implementation for face verification with real-world deployment features
- **Key Adaptation**: Integration with modern face detection (MTCNN), video-based registration, and secure user management
- **Practical Enhancement**: Local deployment with contamination detection and dual security modes

The result is a practical, deployable face verification system that maintains the theoretical rigor of the original research while adding real-world usability and security features.

## 🚀 Quick Start

```bash
# Clone the repository
git clone <your-repo-url>
cd FaceID

# Navigate to the web application
cd FaceID_WebApp

# Option 1: Use automated installation script (Recommended)
chmod +x install_deps.sh
./install_deps.sh

# Option 2: Use setup.py for automated setup
python setup.py install

# Option 3: Manual installation
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

## ✨ Features

- 🔐 **Advanced Face Verification** - Siamese neural network with 95%+ accuracy
- 👤 **Video-based Registration** - Automatic face extraction from uploaded videos
- 📱 **Real-time Verification** - Webcam and image upload support
- 🛡️ **Security Features** - Contamination detection and strict verification modes
- 🎨 **Modern UI** - Professional, responsive Streamlit interface
- 🏠 **Privacy-First** - All processing happens locally, no external API calls
- 📓 **Complete Training Pipeline** - Jupyter notebook with model development and training code
- 📊 **Performance Metrics** - Built-in precision and recall tracking

## 📁 Project Structure

```
FaceID/
├── README.md                                       # Project overview and setup guide
├── LICENSE                                         # MIT license
├── .gitignore                                      # Git ignore rules
├── Siamese Neural Networks for One-shot Image Recognition.pdf # Original research paper
├── Siamese_Network.ipynb                           # Training notebook with model development
├── models/                                         # AI models and checkpoints
│   ├── siamese_network.h5                          # Pre-trained Siamese model
│   └── training_checkpoints/                       # Model training checkpoints
└── FaceID_WebApp/                                  # Web application for inference
    ├── app.py                                      # Main Streamlit application
    ├── requirements.txt                            # Python dependencies
    ├── setup.py                                    # Automated installation and setup
    ├── install_deps.sh                             # Shell script for dependency installation
    ├── utils/                                      # Core utilities
    │   ├── __init__.py                             # Package initialization
    │   ├── face_detection.py                       # MTCNN face detection
    │   ├── siamese_model.py                        # Model architecture & loading
    │   ├── verification.py                         # Face comparison logic
    │   └── face_detection_opencv.py                # Fallback detection with OpenCV
    └── user_data/                                  # Local user face data (excluded from git)
        └── <username>/                             # Individual user directories
```

## 🛠️ Technology Stack

- **Machine Learning**: TensorFlow 2.x, Siamese Neural Networks
- **Face Detection**: MTCNN (Multi-task CNN)
- **Web Interface**: Streamlit
- **Image Processing**: OpenCV, PIL
- **Backend**: Python 3.8+
- **Deployment**: Local installation only

## 📊 Model Performance

- **Architecture**: Siamese Neural Network with L1 Distance layer
- **Training Dataset**: LFW (Labeled Faces in the Wild) + Custom data
- **Accuracy**: 95%+ on face verification tasks
- **Input Size**: 105x105 RGB images
- **Speed**: <0.5s per verification
- **Security**: Advanced contamination detection algorithms

## 🔧 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (for live verification)
- 2GB+ RAM (for model loading)
- Git (for cloning the repository)

### Method 1: Automated Installation Script (Recommended)
```bash
cd FaceID_WebApp
chmod +x install_deps.sh
./install_deps.sh
```
This script will:
- Create a virtual environment
- Install all required dependencies
- Set up the project structure
- Download any missing model files

### Method 2: Setup.py Installation
```bash
cd FaceID_WebApp
python setup.py install
```
This will automatically handle all dependencies and setup.

### Method 3: Manual Installation
```bash
cd FaceID_WebApp
pip install -r requirements.txt
streamlit run app.py
```

### Method 4: Virtual Environment (Recommended for development)
```bash
cd FaceID_WebApp
python -m venv faceid_env
source faceid_env/bin/activate  # On Windows: faceid_env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## 📖 Usage Guide

### 1. **User Registration (Web App)**
- Navigate to "Register" page
- Enter a unique username
- Upload a 5-10 second video of your face
- Turn your head and change expressions
- System automatically extracts and saves multiple face images

### 2. **Face Verification (Web App)**
- Go to "Verify" page
- Select registered username
- Choose verification method:
  - **Upload Image**: Select a photo file
  - **Live Webcam**: Take photo with camera
- Get instant verification results with confidence scores

### 3. **User Management**
- View all registered users
- See face image counts and statistics
- Delete users with confirmation prompts
- View detailed user information

### 4. **System Settings**
- Adjust verification thresholds
- Switch between Strict/Balanced security modes
- View system status and diagnostics

## 🔐 Security Features

### **Contamination Detection**
- Detects suspicious perfect match rates
- Monitors average confidence levels
- Prevents data poisoning attacks
- Flags potential security breaches

### **Dual Security Modes**
- **Strict Mode**: Maximum security with full contamination detection
- **Balanced Mode**: User-friendly with essential security checks

### **Privacy Protection**
- All face data stored locally
- No external API calls or data transmission
- User data excluded from version control
- Complete offline operation

## 🔍 Model Development & Training

The `Siamese_Network.ipynb` notebook contains the complete implementation and training of the Siamese neural network based on the research by Koch et al. (2015). **New users do NOT need to run this notebook** - the pre-trained model is already included and ready to use.

### Model Implementation Details in Notebook:
1. **Data Collection Process**: Custom face data collection using webcam
2. **Data Preprocessing**: Face detection, resizing to 105x105, and normalization
3. **Model Architecture**: Siamese network with L1 distance layer (based on Koch et al.)
4. **Training Process**: Custom training loop with Adam optimizer and binary crossentropy loss
5. **Evaluation Results**: Model performance metrics including precision, recall, and accuracy
6. **Model Saving**: Final trained model saved for inference

### Research Paper Implementation:
- **Architecture**: Faithful implementation of the Siamese CNN described in the original paper
- **Distance Function**: L1 distance layer as specified in Koch et al. research
- **Training Methodology**: One-shot learning approach for face verification
- **Loss Function**: Binary crossentropy for similarity prediction

**Note**: The model has been trained and saved as `models/siamese_network.h5` - you can directly use the web application without any training!

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Original Research**: "Siamese Neural Networks for One-shot Image Recognition" by Gregory Koch, Richard Zemel, and Ruslan Salakhutdinov (2015)
  - This project implements the Siamese neural network architecture described in their groundbreaking paper
  - The L1 distance layer and training methodology are based on their research
  - Paper included in repository: `Siamese Neural Networks for One-shot Image Recognition.pdf`
- **Face Detection**: MTCNN implementation for accurate face detection
- **Framework**: TensorFlow and Streamlit development teams
- **Dataset**: LFW (Labeled Faces in the Wild) database for training negative samples

## 🆘 Support & FAQ

### Common Issues

**Q: Model loading fails**
A: Ensure you have the `siamese_network.h5` file or training checkpoints in the project root

**Q: Face detection not working**
A: The system will automatically fallback from MTCNN to OpenCV. Ensure good lighting conditions.

**Q: Installation fails**
A: Try using the `install_deps.sh` script or create a virtual environment first

**Q: High memory usage**
A: The TensorFlow model requires ~2GB RAM. Close other applications if needed.

**Q: Webcam not detected**
A: Ensure your webcam is connected and not being used by other applications

### Getting Help

- 📖 Check the model development notebook for implementation details
- 🐛 Report bugs via GitHub Issues
- 💡 Request features via GitHub Discussions

---

**🔐 Secure • 🚀 Fast • 🛡️ Private • 💻 Local**

Built with ❤️ for secure local face verification applications.
