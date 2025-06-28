import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time

# Import custom modules with fallback
try:
    from utils.face_detection import FaceDetector
    print("Using MTCNN face detection")
except ImportError as e:
    print(f"MTCNN not available: {e}")
    print("Falling back to OpenCV face detection...")
    try:
        from utils.face_detection_opencv import FaceDetector
        print("Using OpenCV face detection")
    except ImportError as e2:
        print(f"OpenCV face detection also failed: {e2}")
        st.error("Face detection modules not available. Please install required dependencies.")
        st.stop()

from utils.siamese_model import load_siamese_model
from utils.verification import FaceVerifier

# Page configuration
st.set_page_config(
    page_title="FaceID - Face Verification System",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Dhruv-2004/Siamese-FaceID',
        'Report a bug': 'https://github.com/Dhruv-2004/Siamese-FaceID/issues',
        'About': '''
        # FaceID v2.0
        Advanced face verification system powered by Siamese Neural Networks.
        
        **Features:**
        - Real-time face detection & verification
        - Secure local data storage
        - Advanced contamination detection
        - Modern responsive UI
        
        Built with ❤️ using Streamlit & TensorFlow
        '''
    }
)

# Enhanced Custom CSS for modern UI
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styles */
.stApp {
    font-family: 'Inter', sans-serif;
}

/* Header Styles */
.main-header {
    font-size: 3.5rem;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.section-header {
    font-size: 2rem;
    font-weight: 600;
    color: #22c55e;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
    border-bottom: 3px solid #e5e7eb;
    padding-bottom: 0.5rem;
}

/* Enhanced Message Styles */
.success-message {
    background: linear-gradient(135deg, #d1fae5 0%, #ecfdf5 100%);
    border: 1px solid #10b981;
    border-left: 5px solid #10b981;
    color: #065f46;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
}

.error-message {
    background: linear-gradient(135deg, #fee2e2 0%, #fef2f2 100%);
    border: 1px solid #ef4444;
    border-left: 5px solid #ef4444;
    color: #991b1b;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
}

.info-message {
    background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
    border: 1px solid #3b82f6;
    border-left: 5px solid #3b82f6;
    color: #1e40af;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
}

.warning-message {
    background: linear-gradient(135deg, #fef3c7 0%, #fffbeb 100%);
    border: 1px solid #f59e0b;
    border-left: 5px solid #f59e0b;
    color: #92400e;
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    animation: slideIn 0.3s ease-out;
}

/* Feature Cards */
.feature-card {
    background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
}

.feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Statistics Cards */
.stat-card {
    background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: all 0.3s ease;
}

.stat-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

/* Animations */
@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

.fade-in {
    animation: fadeIn 0.5s ease-in;
}

/* Custom Button Styles */
.stButton > button {
    border-radius: 8px;
    font-weight: 500;
    transition: all 0.3s ease;
    border: none;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
}

/* Sidebar Styling */
.css-1d391kg {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

/* Progress Bar Styling */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display:none;}

/* User card styling */
.user-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    transition: all 0.3s ease;
}

.user-card:hover {
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

/* Professional Verification result styling */
.verification-result {
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1rem 0;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    border-left: 4px solid;
}

.verification-success {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    border-color: #16a34a;
    color: #15803d;
}

.verification-failure {
    background: linear-gradient(135deg, #fef2f2 0%, #fee2e2 100%);
    border-color: #dc2626;
    color: #dc2626;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the Siamese model (cached for performance)"""
    try:
        model_path = "../models/siamese_network.h5"
        if not os.path.exists(model_path):
            # Try loading from checkpoint
            checkpoint_path = "models/training_checkpoints/siamese_ckpt.weights.h5"
            if os.path.exists(checkpoint_path):
                st.info("Loading model from checkpoint...")
                model = load_siamese_model(checkpoint_path)
            else:
                st.error("No trained model found! Please ensure model files are in the correct location.")
                return None
        else:
            model = load_siamese_model(model_path)
        
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_resource
def initialize_components():
    """Initialize face detector and load model"""
    face_detector = FaceDetector()
    model = load_model()
    
    if model is not None:
        verifier = FaceVerifier(model, detection_threshold=0.3, verification_threshold=0.3, strict_mode=True)
        return face_detector, verifier
    else:
        return face_detector, None

def main():
    # Initialize components
    face_detector, verifier = initialize_components()
    
    if verifier is None:
        st.error("Failed to initialize the verification system. Please check the model files.")
        return
    
    # Main title
    st.markdown('<h1 class="main-header">🔐 FaceID - Face Verification System</h1>', unsafe_allow_html=True)
    
    # Enhanced sidebar navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 1rem;">
            <h2 style="color: white; margin: 0; font-size: 1.5rem;">🔐 FaceID</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 0.5rem 0 0 0; font-size: 0.9rem;">Neural Face Verification</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation with icons
        st.markdown("### 🧭 Navigation")
        page = st.selectbox(
            "Choose a page", 
            ["🏠 Home", "📝 Register", "🔍 Verify", "👥 User Management", "⚙️ Settings"],
            format_func=lambda x: x.split(' ', 1)[1] if ' ' in x else x
        )
        
        # Map display names back to simple names
        page_mapping = {
            "🏠 Home": "Home",
            "📝 Register": "Register", 
            "🔍 Verify": "Verify",
            "👥 User Management": "User Management",
            "⚙️ Settings": "Settings"
        }
        page = page_mapping.get(page, page)
        
        # System status
        st.markdown("---")
        st.markdown("### 📊 System Status")
        
        # Model status
        model_status = "🟢 Ready" if verifier else "🔴 Error"
        st.markdown(f"**Model:** {model_status}")
        
        # User count
        try:
            user_count = len(verifier.get_user_list()) if verifier else 0
            st.markdown(f"**Users:** {user_count}")
        except:
            st.markdown("**Users:** N/A")
        
        # Security mode
        try:
            mode = "🔒 Strict" if verifier and verifier.strict_mode else "⚖️ Balanced"
            st.markdown(f"**Mode:** {mode}")
        except:
            st.markdown("**Mode:** Unknown")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: rgba(255,255,255,0.6); font-size: 0.8rem;">
            <p>v2.0 | Built with ❤️ by Dhruv Suvagiya</p>
            <p>🛡️ Privacy-First Design</p>
        </div>
        """, unsafe_allow_html=True)
    
    if page == "Home":
        show_home_page()
    elif page == "Register":
        show_registration_page(face_detector)
    elif page == "Verify":
        show_verification_page(face_detector, verifier)
    elif page == "User Management":
        show_user_management_page(verifier)
    elif page == "Settings":
        show_settings_page(verifier)

def show_home_page():
    """Display the enhanced home page with system information"""
    st.markdown('<div class="fade-in">', unsafe_allow_html=True)
    
    # Hero section with enhanced styling
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h2 style="font-size: 2.5rem; font-weight: 600; color: #22c55e; margin-bottom: 1rem;">
            Welcome to FaceID ✨
        </h2>
        <p style="font-size: 1.2rem; color: #6b7280; margin-bottom: 2rem;">
            Advanced face verification powered by Siamese Neural Networks
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; font-size: 2rem; margin: 0;">99%</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #10b981; font-size: 2rem; margin: 0;"><0.5s</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Verification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #f59e0b; font-size: 2rem; margin: 0;">100%</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Local</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #ef4444; font-size: 2rem; margin: 0;">🔒</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Secure</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Feature cards
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-right: 1rem;">🎥</div>
                <h3 style="color: #22c55e; margin: 0;">User Registration</h3>
            </div>
            <ul style="color: #6b7280; line-height: 1.6;">
                <li>🚀 <strong>Quick video upload</strong> - Just 5-10 seconds needed</li>
                <li>🤖 <strong>Automatic face extraction</strong> - AI finds and crops faces</li>
                <li>📸 <strong>Multiple angles</strong> - Turn head for better coverage</li>
                <li>✅ <strong>Quality validation</strong> - Ensures good training data</li>
            </ul>
            <div style="margin-top: 1.5rem;">
                <small style="color: #9ca3af;">💡 Tip: Good lighting and clear view improve accuracy</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                <div style="font-size: 3rem; margin-right: 1rem;">🔍</div>
                <h3 style="color: #22c55e; margin: 0;">Face Verification</h3>
            </div>
            <ul style="color: #6b7280; line-height: 1.6;">
                <li>📱 <strong>Live webcam</strong> - Real-time verification</li>
                <li>📁 <strong>Image upload</strong> - Support for JPG/PNG files</li>
                <li>⚡ <strong>Instant results</strong> - Get confidence scores immediately</li>
                <li>🛡️ <strong>Anti-spoofing</strong> - Advanced security measures</li>
            </ul>
            <div style="margin-top: 1.5rem;">
                <small style="color: #9ca3af;">🔒 Privacy: All processing happens locally</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Technology showcase
    st.markdown("""
    <div style="margin: 3rem 0;">
        <h3 style="text-align: center; color: #22c55e; margin-bottom: 2rem;">🧠 Powered by Advanced AI</h3>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 16px; padding: 2rem; color: white; text-align: center;">
            <div style="display: flex; justify-content: space-around; flex-wrap: wrap; gap: 2rem;">
                <div>
                    <h4 style="margin: 0 0 0.5rem 0;">🧪 Siamese Networks</h4>
                    <p style="margin: 0; opacity: 0.9;">One-shot learning for face comparison</p>
                </div>
                <div>
                    <h4 style="margin: 0 0 0.5rem 0;">👁️ MTCNN Detection</h4>
                    <p style="margin: 0; opacity: 0.9;">Multi-task CNN for face detection</p>
                </div>
                <div>
                    <h4 style="margin: 0 0 0.5rem 0;">🛡️ Contamination Detection</h4>
                    <p style="margin: 0; opacity: 0.9;">Advanced security measures</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick start section
    st.markdown("""
    <div class="info-message" style="text-align: center;">
        <h4 style="margin: 0 0 1rem 0; color: #1e40af;">🚀 Ready to Get Started?</h4>
        <p style="margin: 0;">
            Navigate to <strong>Register</strong> to add your face, then use <strong>Verify</strong> to test the system!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_registration_page(face_detector):
    """Display the user registration page"""
    st.markdown('<h2 class="section-header">👤 User Registration</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-message">
    Register yourself by uploading a short video of your face. For best results:
    <ul>
    <li>Record for 5-10 seconds</li>
    <li>Turn your head left and right</li>
    <li>Ensure good lighting</li>
    <li>Look directly at the camera</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Username input
    username = st.text_input("Enter Username", placeholder="e.g., john_doe")
    
    if username:
        # Check if user already exists
        user_folder = os.path.join("user_data", username)
        if os.path.exists(user_folder):
            st.warning(f"User '{username}' already exists! Use a different username.")
            return
        
        # Video upload
        uploaded_file = st.file_uploader(
            "Upload Registration Video", 
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a short video of your face (5-10 seconds recommended)"
        )
        
        if uploaded_file is not None:
            # Display video
            st.video(uploaded_file)
            
            if st.button("Process Registration Video", type="primary"):
                with st.spinner("Processing video and extracting faces..."):
                    # Save uploaded video temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        temp_video_path = tmp_file.name
                    
                    try:
                        # Process video for registration
                        success, message, face_paths = face_detector.process_video_for_registration(
                            temp_video_path, username
                        )
                        
                        if success:
                            st.markdown(f'<div class="success-message">{message}</div>', unsafe_allow_html=True)
                            
                            # Display extracted faces
                            st.markdown("### Extracted Face Images:")
                            cols = st.columns(min(len(face_paths), 5))
                            
                            for i, face_path in enumerate(face_paths[:5]):  # Show first 5 faces
                                with cols[i % 5]:
                                    img = Image.open(face_path)
                                    st.image(img, caption=f"Face {i+1}", use_container_width=True)
                            
                            if len(face_paths) > 5:
                                st.info(f"Total {len(face_paths)} faces extracted. Showing first 5.")
                            
                            st.success("Registration completed successfully! You can now go to the Verification page.")
                        
                        else:
                            st.markdown(f'<div class="error-message">{message}</div>', unsafe_allow_html=True)
                    
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_video_path):
                            os.unlink(temp_video_path)

def show_verification_page(face_detector, verifier):
    """Display the face verification page"""
    st.markdown('<h2 class="section-header">🔍 Face Verification</h2>', unsafe_allow_html=True)
    
    # Get list of registered users
    users = verifier.get_user_list()
    
    if not users:
        st.warning("No registered users found. Please register first!")
        return
    
    # Username selection
    username = st.selectbox("Select Username", users)
    
    if username:
        user_info = verifier.get_user_info(username)
        if user_info:
            st.info(f"User: {username} | Registered faces: {user_info['num_faces']}")
        
        # Verification method selection
        verification_method = st.radio(
            "Choose verification method:",
            ["Upload Image", "Use Webcam (Live)"]
        )
        
        if verification_method == "Upload Image":
            uploaded_image = st.file_uploader(
                "Upload your photo for verification",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear photo of your face"
            )
            
            if uploaded_image is not None:
                # Display uploaded image
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", width=300)
                
                if st.button("Verify Face", type="primary"):
                    with st.spinner("Processing and verifying face..."):
                        # Convert PIL to numpy array
                        image_array = np.array(image)
                        
                        # Process image for verification
                        success, result = face_detector.process_image_for_verification(image_array)
                        
                        if success:
                            # Verify face
                            is_verified, confidence, details = verifier.verify_face(result, username)
                            
                            # Display enhanced results
                            if is_verified:
                                st.markdown(f'''
                                <div class="verification-result verification-success">
                                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                        <div style="width: 40px; height: 40px; background: #16a34a; border-radius: 50%; 
                                                    display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                            <span style="color: white; font-size: 1.2rem; font-weight: bold;">✓</span>
                                        </div>
                                        <div>
                                            <h3 style="margin: 0; color: #15803d; font-size: 1.4rem;">Verification Successful</h3>
                                            <p style="margin: 0; color: #166534; opacity: 0.8;">Identity confirmed</p>
                                        </div>
                                    </div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; 
                                                background: rgba(255,255,255,0.6); padding: 1rem; border-radius: 6px;">
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">User</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">{username}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">Confidence</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">{confidence:.1%}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">Status</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">Authenticated</div>
                                        </div>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # Professional result display
                                # Professional verification - no animations
                            else:
                                st.markdown(f'''
                                <div class="verification-result verification-failure">
                                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                        <div style="width: 40px; height: 40px; background: #dc2626; border-radius: 50%; 
                                                    display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                            <span style="color: white; font-size: 1.2rem; font-weight: bold;">✕</span>
                                        </div>
                                        <div>
                                            <h3 style="margin: 0; color: #dc2626; font-size: 1.4rem;">Verification Failed</h3>
                                            <p style="margin: 0; color: #991b1b; opacity: 0.8;">Identity not confirmed</p>
                                        </div>
                                    </div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; 
                                                background: rgba(255,255,255,0.6); padding: 1rem; border-radius: 6px;">
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">User</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">{username}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">Best Score</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">{confidence:.1%}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">Status</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">Rejected</div>
                                        </div>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f'<div class="error-message">{result}</div>', unsafe_allow_html=True)
        
        elif verification_method == "Use Webcam (Live)":
            st.markdown("""
            <div class="info-message">
            Position your face clearly in front of the camera and click "Take Photo".
            </div>
            """, unsafe_allow_html=True)
            
            # Webcam capture
            captured_image = st.camera_input("Take a photo for verification")
            
            if captured_image is not None:
                if st.button("Verify Captured Image", type="primary"):
                    with st.spinner("Processing and verifying face..."):
                        # Convert captured image to array
                        image = Image.open(captured_image)
                        image_array = np.array(image)
                        
                        # Process image for verification
                        success, result = face_detector.process_image_for_verification(image_array)
                        
                        if success:
                            # Verify face
                            is_verified, confidence, details = verifier.verify_face(result, username)
                            
                            # Display enhanced results
                            if is_verified:
                                st.markdown(f'''
                                <div class="verification-result verification-success">
                                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                        <div style="width: 40px; height: 40px; background: #16a34a; border-radius: 50%; 
                                                    display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                            <span style="color: white; font-size: 1.2rem; font-weight: bold;">✓</span>
                                        </div>
                                        <div>
                                            <h3 style="margin: 0; color: #15803d; font-size: 1.4rem;">Verification Successful</h3>
                                            <p style="margin: 0; color: #166534; opacity: 0.8;">Identity confirmed</p>
                                        </div>
                                    </div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; 
                                                background: rgba(255,255,255,0.6); padding: 1rem; border-radius: 6px;">
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">User</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">{username}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">Confidence</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">{confidence:.1%}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #166534; font-weight: 500;">Status</div>
                                            <div style="font-size: 1.1rem; color: #15803d; font-weight: 600;">Authenticated</div>
                                        </div>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)
                                
                                # Professional result display
                                # Professional verification - no animations
                            else:
                                st.markdown(f'''
                                <div class="verification-result verification-failure">
                                    <div style="display: flex; align-items: center; margin-bottom: 1rem;">
                                        <div style="width: 40px; height: 40px; background: #dc2626; border-radius: 50%; 
                                                    display: flex; align-items: center; justify-content: center; margin-right: 1rem;">
                                            <span style="color: white; font-size: 1.2rem; font-weight: bold;">✕</span>
                                        </div>
                                        <div>
                                            <h3 style="margin: 0; color: #dc2626; font-size: 1.4rem;">Verification Failed</h3>
                                            <p style="margin: 0; color: #991b1b; opacity: 0.8;">Identity not confirmed</p>
                                        </div>
                                    </div>
                                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; 
                                                background: rgba(255,255,255,0.6); padding: 1rem; border-radius: 6px;">
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">User</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">{username}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">Best Score</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">{confidence:.1%}</div>
                                        </div>
                                        <div style="text-align: center;">
                                            <div style="font-size: 0.9rem; color: #991b1b; font-weight: 500;">Status</div>
                                            <div style="font-size: 1.1rem; color: #dc2626; font-weight: 600;">Rejected</div>
                                        </div>
                                    </div>
                                </div>
                                ''', unsafe_allow_html=True)
                        
                        else:
                            st.markdown(f'<div class="error-message">{result}</div>', unsafe_allow_html=True)

def show_user_management_page(verifier):
    """Display the enhanced user management page"""
    st.markdown('<h2 class="section-header">👥 User Management</h2>', unsafe_allow_html=True)
    
    users = verifier.get_user_list()
    
    if not users:
        st.markdown("""
        <div class="info-message" style="text-align: center;">
            <h4 style="margin: 0 0 1rem 0;">No Users Registered Yet</h4>
            <p style="margin: 0;">Head over to the <strong>Register</strong> page to add your first user!</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Summary stats
    total_faces = sum(verifier.get_user_info(user).get('num_faces', 0) for user in users)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #667eea; font-size: 2rem; margin: 0;">{}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Total Users</p>
        </div>
        """.format(len(users)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #10b981; font-size: 2rem; margin: 0;">{}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Face Images</p>
        </div>
        """.format(total_faces), unsafe_allow_html=True)
    
    with col3:
        avg_faces = total_faces / len(users) if users else 0
        st.markdown("""
        <div class="stat-card">
            <h3 style="color: #f59e0b; font-size: 2rem; margin: 0;">{:.1f}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Avg per User</p>
        </div>
        """.format(avg_faces), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display user information with enhanced cards
    for i, username in enumerate(users):
        user_info = verifier.get_user_info(username)
        
        if user_info:
            # User card with modern styling
            st.markdown(f"""
            <div class="user-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0 0 0.5rem 0; color: #1f2937; display: flex; align-items: center;">
                            <span style="font-size: 1.5rem; margin-right: 0.5rem;">👤</span>
                            {user_info['username']}
                        </h4>
                        <div style="display: flex; gap: 1rem; color: #6b7280;">
                            <span>📸 {user_info['num_faces']} faces</span>
                            <span>🆔 User #{i+1}</span>
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons in columns
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col2:
                if st.button(f"🗑️ Delete", key=f"delete_{username}", 
                           help=f"Delete user {username} and all associated data"):
                    # Add confirmation dialog
                    st.session_state[f"confirm_delete_{username}"] = True
            
            with col3:
                if st.button(f"ℹ️ Details", key=f"details_{username}",
                           help=f"View detailed information for {username}"):
                    st.session_state[f"show_details_{username}"] = not st.session_state.get(f"show_details_{username}", False)
            
            # Confirmation dialog for deletion
            if st.session_state.get(f"confirm_delete_{username}", False):
                st.markdown("""
                <div class="warning-message">
                    <strong>⚠️ Confirm Deletion</strong><br>
                    This action cannot be undone. All face data for this user will be permanently deleted.
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col2:
                    if st.button("✅ Confirm", key=f"confirm_yes_{username}", type="primary"):
                        success, message = verifier.delete_user(username)
                        if success:
                            st.success(message)
                            # Clear the confirmation state
                            del st.session_state[f"confirm_delete_{username}"]
                            st.rerun()
                        else:
                            st.error(message)
                
                with col3:
                    if st.button("❌ Cancel", key=f"confirm_no_{username}"):
                        # Clear the confirmation state
                        del st.session_state[f"confirm_delete_{username}"]
                        st.rerun()
            
            # Detailed information panel
            if st.session_state.get(f"show_details_{username}", False):
                with st.expander(f"📊 Detailed Information for {username}", expanded=True):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**User Information:**")
                        st.write(f"• Username: `{user_info['username']}`")
                        st.write(f"• Number of face images: `{user_info['num_faces']}`")
                        
                        # Show some sample images if available
                        import glob
                        user_images = glob.glob(f"user_data/{username}/*.jpg")
                        if user_images:
                            st.write(f"• Sample images:")
                            cols = st.columns(min(3, len(user_images)))
                            for i, img_path in enumerate(user_images[:3]):
                                with cols[i]:
                                    try:
                                        img = Image.open(img_path)
                                        st.image(img, caption=f"Face {i+1}", use_container_width=True)
                                    except:
                                        st.write("Image not available")
                    
                    with col2:
                        st.write("**System Information:**")
                        try:
                            import os
                            user_folder = f"user_data/{username}"
                            if os.path.exists(user_folder):
                                folder_size = sum(os.path.getsize(os.path.join(user_folder, f)) 
                                                for f in os.listdir(user_folder))
                                st.write(f"• Storage used: `{folder_size/1024:.1f} KB`")
                                
                                # Get creation time of folder
                                import time
                                creation_time = os.path.getctime(user_folder)
                                creation_date = time.strftime('%Y-%m-%d %H:%M', time.localtime(creation_time))
                                st.write(f"• Registered: `{creation_date}`")
                        except:
                            st.write("• System info unavailable")
            
            st.markdown("<br>", unsafe_allow_html=True)

def show_settings_page(verifier):
    """Display the settings page"""
    st.markdown('<h2 class="section-header">⚙️ Settings</h2>', unsafe_allow_html=True)
    
    st.markdown("### Verification Mode")
    
    # Handle legacy verifier objects that might not have strict_mode attribute
    try:
        current_mode = "Strict (Recommended)" if verifier.strict_mode else "Balanced"
    except AttributeError:
        # Verifier doesn't have strict_mode attribute, add it
        verifier.strict_mode = True  # Default to strict mode for safety
        current_mode = "Strict (Recommended)"
        st.info("⚠️ Verifier updated to support new security modes. Please refresh the page if needed.")
    
    verification_mode = st.radio(
        "Select verification security level:",
        ["Strict (Recommended)", "Balanced"],
        index=0 if verifier.strict_mode else 1,
        help="Strict mode prevents false positives but may be more restrictive. Balanced mode is more user-friendly."
    )
    
    if verification_mode != current_mode:
        new_strict_mode = verification_mode == "Strict (Recommended)"
        verifier.strict_mode = new_strict_mode
        st.success(f"Verification mode updated to: {verification_mode}")
        
        if new_strict_mode:
            st.info("🔒 **Strict Mode**: Maximum security with contamination detection. May reject some legitimate users with poor-quality data.")
        else:
            st.warning("⚖️ **Balanced Mode**: More user-friendly but with reduced security. Only severe contamination is detected.")
    
    st.markdown("### Verification Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        detection_threshold = st.slider(
            "Detection Threshold",
            min_value=0.1,
            max_value=0.9,
            value=verifier.detection_threshold,
            step=0.05,
            help="Higher values make individual comparisons more strict"
        )
    
    with col2:
        verification_threshold = st.slider(
            "Verification Threshold",
            min_value=0.1,
            max_value=0.9,
            value=verifier.verification_threshold,
            step=0.05,
            help="Higher values require more positive matches for verification"
        )
    
    if st.button("Update Thresholds"):
        verifier.update_thresholds(detection_threshold, verification_threshold)
        st.success("Thresholds updated successfully!")
    
    # Display current settings summary
    st.markdown("### Current Settings Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Detection Threshold", f"{verifier.detection_threshold:.2f}")
    
    with col2:
        st.metric("Verification Threshold", f"{verifier.verification_threshold:.2f}")
    
    with col3:
        mode_emoji = "🔒" if verifier.strict_mode else "⚖️"
        mode_text = "Strict" if verifier.strict_mode else "Balanced"
        st.metric("Security Mode", f"{mode_emoji} {mode_text}")
    
    # Security recommendations
    st.markdown("### 🔐 Security Recommendations")
    
    if not verifier.strict_mode:
        st.warning("""
        **⚠️ Balanced Mode Active**: While more user-friendly, this mode has reduced security.
        Consider switching to Strict Mode if security is critical.
        """)
    else:
                 st.success("""
        **✅ Strict Mode Active**: Maximum security with contamination detection enabled.
        This mode may be more restrictive but provides the highest security level.
        """)
    
    # Troubleshooting section
    st.markdown("### 🔧 Troubleshooting")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Clear Cache & Restart", help="Clear cached models and restart components"):
            # Clear Streamlit cache
            st.cache_resource.clear()
            st.success("Cache cleared! The page will reload automatically.")
            st.rerun()
    
    with col2:
        if st.button("📊 Show System Info", help="Display current system information"):
            st.write("**Verifier Attributes:**")
            verifier_attrs = [attr for attr in dir(verifier) if not attr.startswith('_')]
            st.write(f"Available attributes: {', '.join(verifier_attrs)}")
            st.write(f"Has strict_mode: {hasattr(verifier, 'strict_mode')}")
            if hasattr(verifier, 'strict_mode'):
                st.write(f"Current strict_mode: {verifier.strict_mode}")

if __name__ == "__main__":
    main()
