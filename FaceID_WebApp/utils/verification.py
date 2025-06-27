import os
import numpy as np
import cv2
from .siamese_model import preprocess_image_array
import tensorflow as tf

class FaceVerifier:
    def __init__(self, model, detection_threshold=0.5, verification_threshold=0.5, strict_mode=True):
        """
        Initialize face verifier
        Args:
            model: trained Siamese model
            detection_threshold: threshold for individual comparisons
            verification_threshold: threshold for overall verification
            strict_mode: whether to use ultra-strict verification (recommended)
        """
        self.model = model
        self.detection_threshold = detection_threshold
        self.verification_threshold = verification_threshold
        self.strict_mode = strict_mode
    
    def verify_face(self, input_face, username):
        """
        Verify a face against registered faces for a user
        Args:
            input_face: numpy array of the face to verify
            username: username to verify against
        Returns:
            tuple (is_verified, confidence_score, results)
        """
        try:
            user_folder = os.path.join("user_data", username)
            
            if not os.path.exists(user_folder):
                return False, 0.0, f"User '{username}' not found."
            
            # Get all registered face images for the user
            registered_faces = []
            face_files = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not face_files:
                return False, 0.0, f"No registered faces found for user '{username}'."
            
            # Load and preprocess registered faces
            for face_file in face_files:
                face_path = os.path.join(user_folder, face_file)
                try:
                    # Read image
                    registered_face = cv2.imread(face_path)
                    if registered_face is not None:
                        # Convert BGR to RGB
                        registered_face = cv2.cvtColor(registered_face, cv2.COLOR_BGR2RGB)
                        # Registered faces are already cropped, just resize to ensure 105x105
                        registered_face = cv2.resize(registered_face, (105, 105))
                        registered_faces.append(registered_face)
                except Exception as e:
                    print(f"Error loading {face_path}: {e}")
                    continue
            
            if not registered_faces:
                return False, 0.0, "Could not load any registered faces."
            
            # Preprocess input face (ensure it's 105x105)
            if input_face.shape[:2] != (105, 105):
                input_face = cv2.resize(input_face, (105, 105))
            input_processed = preprocess_image_array(input_face)
            
            # Compare with each registered face
            results = []
            for i, registered_face in enumerate(registered_faces):
                registered_processed = preprocess_image_array(registered_face)
                
                # Expand dimensions for model input
                input_batch = tf.expand_dims(input_processed, axis=0)
                registered_batch = tf.expand_dims(registered_processed, axis=0)
                
                # Make prediction
                prediction = self.model.predict([input_batch, registered_batch], verbose=0)
                similarity_score = float(prediction[0][0])
                
                results.append(similarity_score)
            
            # Calculate verification metrics
            positive_detections = sum(1 for score in results if score > self.detection_threshold)
            verification_ratio = positive_detections / len(results)
            avg_confidence = np.mean(results)
            max_confidence = np.max(results)
            
            if self.strict_mode:
                # ULTRA-STRICT verification logic to combat data contamination
                # Given the discovered contamination, we need much stricter rules
                
                # Base requirements (all must be met)
                base_criteria = []
                base_criteria.append(verification_ratio > max(self.verification_threshold, 0.6))  # At least 60% positive
                base_criteria.append(avg_confidence > 0.6)  # Average must be above 60%
                base_criteria.append(max_confidence > 0.8)   # At least one strong match
                base_criteria.append(positive_detections >= 3)  # At least 3 positive matches
                
                # Anti-contamination checks
                contamination_flags = 0
                
                # Flag 1: Too many perfect matches (indicates potential data contamination)
                perfect_matches = sum(1 for score in results if score > 0.99)
                if perfect_matches > len(results) * 0.3:  # More than 30% perfect matches
                    contamination_flags += 1
                    print(f"Warning: Suspicious perfect match rate for {username}: {perfect_matches}/{len(results)}")
                
                # Flag 2: Suspiciously high average (indicates possible same-person data)
                if avg_confidence > 0.85:
                    contamination_flags += 1
                    print(f"Warning: Suspiciously high average confidence for {username}: {avg_confidence:.3f}")
                
                # Flag 3: Very consistent scores (indicates potential overfitting)
                score_std = np.std(results)
                if score_std < 0.1 and avg_confidence > 0.7:
                    contamination_flags += 1
                    print(f"Warning: Suspiciously low score variance for {username}: {score_std:.3f}")
                
                # Decision logic
                base_requirements_met = all(base_criteria)
                is_verified = False
                
                if base_requirements_met:
                    if contamination_flags == 0:
                        is_verified = True  # Clean verification
                    elif contamination_flags == 1:
                        # Allow with warning if only one flag
                        is_verified = True
                        print(f"Caution: Verification passed for {username} but with 1 contamination flag")
                    else:
                        # Reject if multiple contamination flags
                        is_verified = False
                        print(f"Verification REJECTED for {username} due to {contamination_flags} contamination flags")
                
                verification_details = {
                    'total_comparisons': len(results),
                    'positive_detections': positive_detections,
                    'verification_ratio': verification_ratio,
                    'average_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'individual_scores': results,
                    'base_requirements_met': base_requirements_met,
                    'contamination_flags': contamination_flags,
                    'perfect_matches': perfect_matches,
                    'score_std': score_std,
                    'strict_mode': True
                }
                
            else:
                # BALANCED mode for better user experience
                # More lenient but still secure
                
                # Balanced requirements
                min_avg_confidence = 0.4
                min_max_confidence = 0.5
                min_verification_ratio = max(self.verification_threshold, 0.4)
                
                # Check for obvious contamination (still important)
                perfect_matches = sum(1 for score in results if score > 0.99)
                contamination_detected = False
                
                # Only flag severe contamination in balanced mode
                if perfect_matches > len(results) * 0.6:  # More than 60% perfect matches
                    contamination_detected = True
                    print(f"Severe contamination detected for {username}: {perfect_matches}/{len(results)} perfect matches")
                
                # Balanced verification logic
                criteria_met = 0
                if verification_ratio > min_verification_ratio:
                    criteria_met += 1
                if avg_confidence > min_avg_confidence:
                    criteria_met += 1
                if max_confidence > min_max_confidence:
                    criteria_met += 1
                
                # Require at least 2 out of 3 criteria, and no severe contamination
                is_verified = (criteria_met >= 2) and not contamination_detected
                
                verification_details = {
                    'total_comparisons': len(results),
                    'positive_detections': positive_detections,
                    'verification_ratio': verification_ratio,
                    'average_confidence': avg_confidence,
                    'max_confidence': max_confidence,
                    'individual_scores': results,
                    'criteria_met': criteria_met,
                    'contamination_detected': contamination_detected,
                    'perfect_matches': perfect_matches,
                    'strict_mode': False
                }
            
            return is_verified, max_confidence, verification_details
            
        except Exception as e:
            return False, 0.0, f"Error during verification: {str(e)}"
    
    def get_user_list(self):
        """Get list of registered users"""
        try:
            if not os.path.exists("user_data"):
                return []
            
            users = []
            for item in os.listdir("user_data"):
                user_path = os.path.join("user_data", item)
                if os.path.isdir(user_path):
                    # Check if user has registered faces
                    face_files = [f for f in os.listdir(user_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    if face_files:
                        users.append(item)
            
            return sorted(users)
            
        except Exception as e:
            print(f"Error getting user list: {e}")
            return []
    
    def get_user_info(self, username):
        """Get information about a registered user"""
        try:
            user_folder = os.path.join("user_data", username)
            
            if not os.path.exists(user_folder):
                return None
            
            face_files = [f for f in os.listdir(user_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            info = {
                'username': username,
                'num_faces': len(face_files),
                'face_files': face_files,
                'registration_date': None  # Could be added by checking file creation time
            }
            
            return info
            
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def delete_user(self, username):
        """Delete a user and all their registered faces"""
        try:
            user_folder = os.path.join("user_data", username)
            
            if not os.path.exists(user_folder):
                return False, f"User '{username}' not found."
            
            # Delete all files in user folder
            for file in os.listdir(user_folder):
                file_path = os.path.join(user_folder, file)
                os.remove(file_path)
            
            # Remove user folder
            os.rmdir(user_folder)
            
            return True, f"User '{username}' deleted successfully."
            
        except Exception as e:
            return False, f"Error deleting user: {str(e)}"
    
    def update_thresholds(self, detection_threshold=None, verification_threshold=None):
        """Update verification thresholds"""
        if detection_threshold is not None:
            self.detection_threshold = detection_threshold
        
        if verification_threshold is not None:
            self.verification_threshold = verification_threshold 