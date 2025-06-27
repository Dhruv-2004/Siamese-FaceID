import cv2
import numpy as np
from mtcnn import MTCNN
import os
from PIL import Image
import tempfile

class FaceDetector:
    def __init__(self):
        """Initialize MTCNN face detector"""
        self.detector = MTCNN()
    
    def detect_face_in_image(self, image):
        """
        Detect and extract face from a single image
        Args:
            image: numpy array representing the image
        Returns:
            cropped face image or None if no face detected
        """
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Detect faces
            result = self.detector.detect_faces(rgb_image)
            
            if result:
                # Get the largest face (highest confidence)
                largest_face = max(result, key=lambda x: x['confidence'])
                
                if largest_face['confidence'] > 0.9:  # Confidence threshold
                    # Extract bounding box
                    x, y, width, height = largest_face['box']
                    
                    # Add some padding around the face
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    width = min(rgb_image.shape[1] - x, width + 2 * padding)
                    height = min(rgb_image.shape[0] - y, height + 2 * padding)
                    
                    # Crop the face
                    face = rgb_image[y:y+height, x:x+width]
                    
                    # Resize to standard size (105x105 for the model)
                    face_resized = cv2.resize(face, (105, 105))
                    
                    return face_resized
            
            return None
            
        except Exception as e:
            print(f"Error detecting face: {e}")
            return None
    
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
                    face = self.detect_face_in_image(frame)
                    
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