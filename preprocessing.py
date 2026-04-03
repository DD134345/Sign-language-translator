import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List


class VideoPreprocessor:
    """Preprocessing pipeline for sign language video input."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 denoise: bool = True,
                 enhance_contrast: bool = True):
        self.target_size = target_size
        self.denoise = denoise
        self.enhance_contrast = enhance_contrast
        
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=True,
            smooth_segmentation=True
        )
        
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame and extract landmarks."""
        frame = self._normalize_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = self.holistic.process(frame_rgb)
        
        landmarks = self._extract_landmarks(results)
        
        return {
            'frame': frame,
            'landmarks': landmarks,
            'results': results
        }
    
    def _normalize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize, denoise, and adjust brightness."""
        frame = cv2.resize(frame, self.target_size)
        
        if self.denoise:
            frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)
        
        if self.enhance_contrast:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            frame = cv2.merge([l, a, b])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        return frame
    
    def _extract_landmarks(self, results) -> dict:
        """Extract hand, pose, and face landmarks."""
        landmarks = {
            'left_hand': [],
            'right_hand': [],
            'pose': [],
            'face': []
        }
        
        if results.left_hand_landmarks:
            for lm in results.left_hand_landmarks.landmark:
                landmarks['left_hand'].append([lm.x, lm.y, lm.z])
        
        if results.right_hand_landmarks:
            for lm in results.right_hand_landmarks.landmark:
                landmarks['right_hand'].append([lm.x, lm.y, lm.z])
        
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark:
                landmarks['pose'].append([lm.x, lm.y, lm.z])
        
        if results.face_landmarks:
            for lm in results.face_landmarks.landmark:
                landmarks['face'].append([lm.x, lm.y, lm.z])
        
        return landmarks
    
    def get_hand_features(self, landmarks: dict) -> np.ndarray:
        """Combine hand landmarks into feature vector."""
        left = np.array(landmarks.get('left_hand', [[0]*3]*21))
        right = np.array(landmarks.get('right_hand', [[0]*3]*21))
        
        combined = np.concatenate([left, right], axis=0)
        return combined.flatten()
    
    def get_full_body_features(self, landmarks: dict) -> np.ndarray:
        """Combine all landmarks into full feature vector."""
        left = np.array(landmarks.get('left_hand', [[0]*3]*21))
        right = np.array(landmarks.get('right_hand', [[0]*3]*21))
        pose = np.array(landmarks.get('pose', [[0]*3]*33))
        face = np.array(landmarks.get('face', [[0]*3]*468))
        
        combined = np.concatenate([left, right, pose, face], axis=0)
        return combined
    
    def close(self):
        """Release resources."""
        self.holistic.close()


class LandmarkProcessor:
    """Process and transform landmarks for model input."""
    
    def __init__(self, num_frames: int = 30):
        self.num_frames = num_frames
        
    def pad_or_truncate(self, landmarks: List[np.ndarray]) -> np.ndarray:
        """Ensure all sequences have the same length."""
        if len(landmarks) >= self.num_frames:
            return np.array(landmarks[:self.num_frames])
        else:
            padding = np.zeros((self.num_frames - len(landmarks), 
                              landmarks[0].shape[0] if len(landmarks) > 0 else 126))
            return np.vstack([np.array(landmarks), padding])
    
    def normalize_coordinates(self, landmarks: np.ndarray, 
                             reference: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize landmarks relative to body center or reference point."""
        if reference is None:
            center = landmarks[:, :3].mean(axis=0)
        else:
            center = reference[:3]
        
        normalized = landmarks.copy()
        normalized[:, :3] -= center
        return normalized
    
    def compute_velocity(self, landmarks: np.ndarray) -> np.ndarray:
        """Compute temporal derivatives (velocity) between frames."""
        if len(landmarks) < 2:
            return np.zeros_like(landmarks)
        
        velocity = np.diff(landmarks, axis=0)
        velocity = np.vstack([velocity[0], velocity])
        return velocity
    
    def augment(self, landmarks: np.ndarray, 
                noise_scale: float = 0.01,
                scale_range: Tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply data augmentation to landmarks."""
        augmented = landmarks.copy()
        
        noise = np.random.randn(*augmented.shape) * noise_scale
        augmented += noise
        
        scale = np.random.uniform(*scale_range)
        augmented *= scale
        
        return augmented