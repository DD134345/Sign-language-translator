import cv2
import numpy as np
import torch
import threading
import time
from collections import deque
from typing import Optional, List

from preprocessing import VideoPreprocessor, LandmarkProcessor
from models import get_model


class RealtimeTranslator:
    """Real-time sign language translator using webcam."""
    
    def __init__(self, 
                 model_path: str,
                 class_labels: List[str],
                 model_type: str = 'transformer',
                 num_frames: int = 30,
                 confidence_threshold: float = 0.7,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.class_labels = class_labels
        self.num_frames = num_frames
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        self.preprocessor = VideoPreprocessor()
        self.landmark_processor = LandmarkProcessor(num_frames=num_frames)
        
        self.model = get_model(
            model_type=model_type,
            num_classes=len(class_labels)
        )
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        self.frame_buffer = deque(maxlen=num_frames)
        self.current_prediction = None
        self.running = False
        self.lock = threading.Lock()
        
        self.cap = None
        
    def start(self, camera_index: int = 0):
        """Start the real-time translation."""
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.running = True
        self.process_thread = threading.Thread(target=self._process_frames)
        self.process_thread.start()
        
        return self
    
    def _process_frames(self):
        """Process frames in background thread."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            result = self.preprocessor.process_frame(frame)
            landmarks = result['landmarks']
            
            hand_features = self.preprocessor.get_hand_features(landmarks)
            
            with self.lock:
                self.frame_buffer.append(hand_features)
                
                if len(self.frame_buffer) >= self.num_frames:
                    sequence = np.array(list(self.frame_buffer))
                    prediction = self._predict(sequence)
                    self.current_prediction = prediction
            
            self._draw_overlay(frame, result['landmarks'])
            
            cv2.imshow('Sign Language Translator', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    def _predict(self, sequence: np.ndarray) -> dict:
        """Predict sign from sequence."""
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(dim=1)
        
        predicted_idx = predicted.item()
        confidence_val = confidence.item()
        
        if confidence_val >= self.confidence_threshold:
            return {
                'label': self.class_labels[predicted_idx],
                'confidence': confidence_val
            }
        return None
    
    def _draw_overlay(self, frame, landmarks):
        """Draw landmarks overlay on frame."""
        if landmarks['left_hand']:
            for point in landmarks['left_hand']:
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        
        if landmarks['right_hand']:
            for point in landmarks['right_hand']:
                x, y = int(point[0] * frame.shape[1]), int(point[1] * frame.shape[0])
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
    
    def get_current_prediction(self) -> Optional[dict]:
        """Get current prediction (thread-safe)."""
        with self.lock:
            return self.current_prediction
    
    def stop(self):
        """Stop the translator."""
        self.running = False
        if self.process_thread:
            self.process_thread.join()
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.preprocessor.close()


class BatchTranslator:
    """Batch processing mode for pre-recorded videos."""
    
    def __init__(self, model_path: str, class_labels: List[str], 
                 model_type: str = 'transformer', device: str = 'cuda'):
        
        self.class_labels = class_labels
        self.device = device
        
        self.model = get_model(model_type=model_type, num_classes=len(class_labels))
        
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(device)
        self.model.eval()
        
        self.preprocessor = VideoPreprocessor()
        self.landmark_processor = LandmarkProcessor()
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> List[dict]:
        """Process a video file and return predictions."""
        cap = cv2.VideoCapture(video_path)
        
        sequences = []
        predictions = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.preprocessor.process_frame(frame)
            hand_features = self.preprocessor.get_hand_features(result['landmarks'])
            sequences.append(hand_features)
            
            if len(sequences) >= self.landmark_processor.num_frames:
                sequence = self.landmark_processor.pad_or_truncate(sequences)
                pred = self._predict(sequence)
                predictions.append(pred)
                sequences = sequences[-5:]
        
        cap.release()
        
        if output_path:
            self._save_predictions(predictions, output_path)
        
        return predictions
    
    def _predict(self, sequence: np.ndarray) -> dict:
        sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(sequence)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(dim=1)
        
        return {
            'label': self.class_labels[predicted.item()],
            'confidence': confidence.item()
        }
    
    def _save_predictions(self, predictions: List[dict], output_path: str):
        import json
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)


def create_translator(model_path: str, class_labels_path: str, **kwargs):
    """Factory function to create a translator."""
    import json
    
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)
    
    return RealtimeTranslator(model_path, class_labels, **kwargs)