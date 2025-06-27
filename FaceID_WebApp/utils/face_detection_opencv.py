import cv2
import numpy as np
import os
from PIL import Image
import tempfile

class FaceDetector:
    def __init__(self):
        """Initialize OpenCV face detector (fallback for MTCNN)"""
        # Load OpenCV's pre-trained face detection model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Alternative: Try to load a better face detection model if available
        try:
            # Try to load DNN face detection model (more accurate)
            self.net = cv2.dnn.readNetFromTensorflow(
                cv2.samples.findFile("opencv_face_detector_uint8.pb"),
                cv2.samples.findFile("opencv_face_detector.pbtxt")
            )
            self.use_dnn = True
            print("✅ Using OpenCV DNN face detector")
        except:
            self.net = None
            self.use_dnn = False
            print("⚠️  Using OpenCV Haar Cascade face detector (basic)")
    
    def detect_face_in_image(self, image):
        """
        Detect and extract face from a single image using OpenCV
        Args:
            image: numpy array representing the image
        Returns:
            cropped face image or None if no face detected
        """
        try:
            # Convert to grayscale for detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                rgb_image = image if image.shape[2] == 3 else cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                gray = image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            if self.use_dnn and self.net is not None:
                # Use DNN face detection
                faces = self._detect_faces_dnn(rgb_image)
            else:
                # Use Haar Cascade detection
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
            
            if len(faces) > 0:
                # Get the largest face
                if self.use_dnn:
                    # DNN returns confidence scores
                    best_face = max(faces, key=lambda x: x[4])  # confidence is at index 4
                    x, y, w, h = best_face[:4]
                else:
                    # Haar cascade - get largest by area
                    areas = [w * h for (x, y, w, h) in faces]
                    max_area_idx = np.argmax(areas)
                    x, y, w, h = faces[max_area_idx]
                
                # Add padding around the face
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(rgb_image.shape[1] - x, w + 2 * padding)
                h = min(rgb_image.shape[0] - y, h + 2 * padding)
                
                # Crop the face
                face = rgb_image[y:y+h, x:x+w]
                
                # Resize to standard size (105x105 for the model)
                face_resized = cv2.resize(face, (105, 105))
                
                return face_resized
            
            return None
            
        except Exception as e:
            print(f"Error detecting face: {e}")
            return None
    
    def _detect_faces_dnn(self, image):
        """Detect faces using OpenCV DNN"""
        try:
            h, w = image.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            
            # Set input to the model
            self.net.setInput(blob)
            
            # Run forward pass
            detections = self.net.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter out weak detections
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    faces.append([x1, y1, x2-x1, y2-y1, confidence])
            
            return faces
            
        except Exception as e:
            print(f"DNN face detection error: {e}")
            return []
    
    def extract_faces_from_video(self, video_path, max_frames=30, skip_frames=5):
        """
        Extract multiple face images from a video
        Args:
            video_path: path to the video file
            max_frames: maximum number of frames to process
            skip_frames: number of frames to skip between extractions
        Returns:
            list of face images
        """
        faces = []
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print("Error: Could not open video")
                return faces
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Skip frames for diversity
                if frame_count % skip_frames == 0:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face = self.detect_face_in_image(frame_rgb)
                    
                    if face is not None:
                        faces.append(face)
                        extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting faces from video: {e}")
        
        return faces
    
    def extract_faces_from_webcam_frames(self, frames):
        """
        Extract faces from a list of webcam frames
        Args:
            frames: list of frame images
        Returns:
            list of face images
        """
        faces = []
        
        for frame in frames:
            face = self.detect_face_in_image(frame)
            if face is not None:
                faces.append(face)
        
        return faces
    
    def save_faces(self, faces, user_folder):
        """
        Save extracted faces to user folder
        Args:
            faces: list of face images
            user_folder: path to user's folder
        Returns:
            list of saved file paths
        """
        saved_paths = []
        
        try:
            # Create user folder if it doesn't exist
            os.makedirs(user_folder, exist_ok=True)
            
            for i, face in enumerate(faces):
                # Generate unique filename
                filename = f"face_{i+1}.jpg"
                filepath = os.path.join(user_folder, filename)
                
                # Convert RGB to BGR for OpenCV saving
                face_bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                
                # Save image
                cv2.imwrite(filepath, face_bgr)
                saved_paths.append(filepath)
            
        except Exception as e:
            print(f"Error saving faces: {e}")
        
        return saved_paths
    
    def process_video_for_registration(self, video_path, username):
        """
        Complete pipeline for processing video during registration
        Args:
            video_path: path to uploaded video
            username: username for the user
        Returns:
            tuple (success, message, face_paths)
        """
        try:
            # Extract faces from video
            faces = self.extract_faces_from_video(video_path)
            
            if len(faces) == 0:
                return False, "No faces detected in the video. Please try again with better lighting.", []
            
            if len(faces) < 3:
                return False, f"Only {len(faces)} faces detected. Please upload a longer video or move your head more.", faces
            
            # Create user folder
            user_folder = os.path.join("user_data", username)
            
            # Save faces
            face_paths = self.save_faces(faces, user_folder)
            
            return True, f"Successfully registered {len(faces)} face images for {username}!", face_paths
            
        except Exception as e:
            return False, f"Error processing video: {str(e)}", []
    
    def process_image_for_verification(self, image):
        """
        Process single image for verification
        Args:
            image: numpy array representing the image
        Returns:
            tuple (success, face_image or error_message)
        """
        try:
            face = self.detect_face_in_image(image)
            
            if face is None:
                return False, "No face detected in the image. Please try again with better lighting and positioning."
            
            return True, face
            
        except Exception as e:
            return False, f"Error processing image: {str(e)}" 