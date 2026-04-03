import sys
import os
from pathlib import Path
import traceback

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QComboBox,
                                  QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
                                  QGroupBox, QFormLayout, QTextEdit, QProgressBar,
                                  QFrame, QSlider)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread
    from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
    PYQT5_AVAILABLE = True
except ImportError:
    PYQT5_AVAILABLE = False

import cv2
import numpy as np
import torch
import json
from collections import deque

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False


def create_model(num_classes=100, input_dim=126):
    """Create the sign language model architecture."""
    from models import TemporalTransformer
    return TemporalTransformer(input_dim=input_dim, num_classes=num_classes)


class LandmarkExtractor(QThread):
    """Background thread for landmark extraction."""
    frame_ready = pyqtSignal(object, object)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = False
        self.cap = None
        self._holistic = None
        
    def start_capture(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            self.error_occurred.emit(f"Cannot open camera {camera_index}")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True
        self.start()
        
    def run(self):
        if not MEDIAPIPE_AVAILABLE:
            self.error_occurred.emit("MediaPipe not available")
            return
            
        try:
            mp_holistic = mp.solutions.holistic
            self._holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True
            )
            
            while self.running:
                if self.cap is None or not self.cap.isOpened():
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self._holistic.process(frame_rgb)
                
                hand_features = self._extract_hands(results)
                
                self.frame_ready.emit(frame, hand_features)
                
        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
        finally:
            if self._holistic:
                self._holistic.close()
        
    def _extract_hands(self, results):
        features = np.zeros(126, dtype=np.float32)
        
        if results.left_hand_landmarks:
            for i, lm in enumerate(results.left_hand_landmarks.landmark[:21]):
                features[i*3] = lm.x
                features[i*3+1] = lm.y
                features[i*3+2] = lm.z
                
        if results.right_hand_landmarks:
            for i, lm in enumerate(results.right_hand_landmarks.landmark[:21]):
                features[63 + i*3] = lm.x
                features[63 + i*3+1] = lm.y
                features[63 + i*3+2] = lm.z
                
        return features
        
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.wait(2000)


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")
        self.setMinimumSize(900, 700)
        
        self.model = None
        self.class_labels = []
        self.frame_buffer = deque(maxlen=30)
        self.translator_thread = None
        self.current_prediction = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        left_panel = self.create_control_panel()
        main_layout.addWidget(left_panel, 1)
        
        right_panel = self.create_video_panel()
        main_layout.addWidget(right_panel, 2)
        
    def create_control_panel(self):
        group = QGroupBox("Controls")
        layout = QVBoxLayout()
        
        self.model_label = QLabel("No model loaded")
        self.model_label.setStyleSheet("color: #888; font: 12px;")
        layout.addWidget(self.model_label)
        
        self.device_label = QLabel(f"Device: {self.device}")
        self.device_label.setStyleSheet("color: #666; font: 10px;")
        layout.addWidget(self.device_label)
        
        btn_layout = QVBoxLayout()
        
        self.btn_load_model = QPushButton("Load Model")
        self.btn_load_model.clicked.connect(self.load_model)
        btn_layout.addWidget(self.btn_load_model)
        
        self.btn_load_labels = QPushButton("Load Class Labels")
        self.btn_load_labels.clicked.connect(self.load_labels)
        btn_layout.addWidget(self.btn_load_labels)
        
        self.btn_start = QPushButton("Start Camera")
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_start.setEnabled(False)
        btn_layout.addWidget(self.btn_start)
        
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_stop.setEnabled(False)
        btn_layout.addWidget(self.btn_stop)
        
        layout.addLayout(btn_layout)
        
        settings_group = QGroupBox("Settings")
        settings_layout = QFormLayout()
        
        self.num_frames_spin = QSpinBox()
        self.num_frames_spin.setRange(10, 100)
        self.num_frames_spin.setValue(30)
        self.num_frames_spin.valueChanged.connect(self.update_settings)
        settings_layout.addRow("Sequence Length:", self.num_frames_spin)
        
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.1, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.7)
        self.confidence_spin.valueChanged.connect(self.update_settings)
        settings_layout.addRow("Confidence Threshold:", self.confidence_spin)
        
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 5)
        self.camera_spin.setValue(0)
        settings_layout.addRow("Camera Index:", self.camera_spin)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        self.prediction_label = QLabel("Prediction: --")
        self.prediction_label.setFont(QFont("Arial", 14, QFont.Bold))
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setStyleSheet("padding: 10px; background: #f0f0f0; border-radius: 5px;")
        layout.addWidget(self.prediction_label)
        
        self.confidence_label = QLabel("Confidence: --")
        self.confidence_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.confidence_label)
        
        layout.addStretch()
        
        group.setLayout(layout)
        return group
        
    def create_video_panel(self):
        group = QGroupBox("Video Feed")
        layout = QVBoxLayout()
        
        self.video_label = QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background: #1a1a1a; color: #666;")
        self.video_label.setText("Camera not active\n\nClick 'Start Camera' to begin")
        
        layout.addWidget(self.video_label)
        
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        group.setLayout(layout)
        return group
        
    def log(self, message):
        self.log_text.append(message)
        
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
            
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            num_classes = len(self.class_labels) if self.class_labels else 100
            
            if 'model_state_dict' in checkpoint:
                self.model = create_model(num_classes=num_classes)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.log(f"Loaded model from {Path(path).name}")
            else:
                self.model = create_model(num_classes=num_classes)
                self.model.load_state_dict(checkpoint)
                self.log(f"Loaded model from {Path(path).name}")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_path = path
            self.model_loaded = True
            self.model_label.setText(f"Model: {Path(path).name}")
            
            if self.class_labels:
                self.btn_start.setEnabled(True)
                
        except Exception as e:
            self.log(f"Error: {str(e)}")
            self.log(traceback.format_exc())
            QMessageBox.warning(self, "Error", f"Failed to load model:\n{str(e)}")
                
    def load_labels(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Labels File", "", "JSON (*.json)")
        if not path:
            return
            
        try:
            with open(path, 'r') as f:
                self.class_labels = json.load(f)
                
            if not isinstance(self.class_labels, list):
                raise ValueError("Labels must be a list")
                
            self.labels_path = path
            self.log(f"Loaded {len(self.class_labels)} classes: {', '.join(self.class_labels[:5])}...")
            
            if hasattr(self, 'model_loaded') and self.model_loaded:
                self.btn_start.setEnabled(True)
                    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load labels: {e}")
                
    def start_camera(self):
        if not MEDIAPIPE_AVAILABLE:
            QMessageBox.warning(self, "Error", "MediaPipe not installed.\nRun: pip install mediapipe")
            return
            
        if self.model is None:
            QMessageBox.warning(self, "Error", "Please load a model first")
            return
            
        if not self.class_labels:
            QMessageBox.warning(self, "Error", "Please load class labels first")
            return
            
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        
        self.frame_buffer = deque(maxlen=self.num_frames_spin.value())
        
        self.translator_thread = LandmarkExtractor()
        self.translator_thread.frame_ready.connect(self.on_frame)
        self.translator_thread.error_occurred.connect(self.on_camera_error)
        self.translator_thread.start_capture(self.camera_spin.value())
        
        self.log("Camera started")
        self.video_label.setText("Processing...")
        
    def on_camera_error(self, error_msg):
        self.log(error_msg)
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", error_msg)
        
    def on_frame(self, frame, hand_features):
        if frame is None:
            return
            
        self.frame_buffer.append(hand_features)
        
        if len(self.frame_buffer) >= self.num_frames_spin.value():
            self.predict()
            
        h, w = frame.shape[:2]
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bytes_per_line = 3 * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        scaled = qt_image.scaled(self.video_label.size(), Qt.KeepAspectRatio)
        self.video_label.setPixmap(QPixmap.fromImage(scaled))
        
    def predict(self):
        if self.model is None or not self.class_labels:
            return
            
        if len(self.frame_buffer) == 0:
            return
            
        try:
            sequence = np.array(list(self.frame_buffer), dtype=np.float32)
            
            if sequence.shape[0] < self.num_frames_spin.value():
                padding = np.zeros((self.num_frames_spin.value() - sequence.shape[0], 126), dtype=np.float32)
                sequence = np.vstack([sequence, padding])
            
            sequence = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(sequence)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = probs.max(dim=1)
                
            if confidence.item() >= self.confidence_spin.value():
                pred_idx = predicted.item()
                if 0 <= pred_idx < len(self.class_labels):
                    self.current_prediction = self.class_labels[pred_idx]
                    self.prediction_label.setText(f"Prediction: {self.current_prediction}")
                    self.confidence_label.setText(f"Confidence: {confidence.item()*100:.1f}%")
        except Exception as e:
            self.log(f"Prediction error: {e}")
        
    def stop_camera(self):
        if self.translator_thread:
            self.translator_thread.stop()
            self.translator_thread = None
            
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        
        self.log("Camera stopped")
        self.video_label.setText("Camera not active\n\nClick 'Start Camera' to begin")
        
    def update_settings(self):
        self.frame_buffer = deque(maxlen=self.num_frames_spin.value())
        
    def load_settings(self):
        settings_file = Path("settings.json")
        if settings_file.exists():
            try:
                with open(settings_file, 'r') as f:
                    settings = json.load(f)
                    self.num_frames_spin.setValue(settings.get('num_frames', 30))
                    self.confidence_spin.setValue(settings.get('confidence', 0.7))
                    self.camera_spin.setValue(settings.get('camera_index', 0))
            except:
                pass
                
    def closeEvent(self, event):
        if self.translator_thread:
            self.translator_thread.stop()
            self.translator_thread = None
            
        settings = {
            'num_frames': self.num_frames_spin.value(),
            'confidence': self.confidence_spin.value(),
            'camera_index': self.camera_spin.value()
        }
        
        try:
            with open('settings.json', 'w') as f:
                json.dump(settings, f)
        except:
            pass
            
        event.accept()


def run_app():
    if not PYQT5_AVAILABLE:
        print("ERROR: PyQt5 not installed.")
        print("Run: pip install PyQt5")
        return
        
    if not MEDIAPIPE_AVAILABLE:
        print("WARNING: MediaPipe not installed.")
        print("Run: pip install mediapipe")
        
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()