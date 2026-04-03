import sys
import os
from pathlib import Path
import traceback
import ast

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                  QHBoxLayout, QPushButton, QLabel, QComboBox,
                                  QSpinBox, QDoubleSpinBox, QFileDialog, QMessageBox,
                                  QGroupBox, QFormLayout, QTextEdit, QProgressBar,
                                  QFrame, QSlider, QTabWidget, QTableWidget,
                                  QTableWidgetItem, QHeaderView)
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

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import openpyxl
    OPENPYXL_AVAILABLE = True
except ImportError:
    OPENPYXL_AVAILABLE = False

DEFAULT_NUM_FRAMES = 30
INPUT_DIM = 126


def create_model(num_classes=100, input_dim=INPUT_DIM):
    """Create the sign language model architecture."""
    from models import TemporalTransformer
    return TemporalTransformer(input_dim=input_dim, num_classes=num_classes)


def load_dataset(file_path):
    """Load dataset from file. Supports JSON, CSV, Excel, NumPy."""
    path = Path(file_path)
    ext = path.suffix.lower()
    
    if ext == '.json':
        return _load_json_dataset(file_path)
    elif ext in ['.csv', '.xlsx', '.xls']:
        return _load_tabular_dataset(file_path, ext)
    elif ext == '.npy':
        return _load_npy_dataset(file_path)
    
    return None, None


def load_dataset_from_dataframe(df, label_column=None):
    """Load dataset from a Pandas DataFrame.
    
    Args:
        df: Pandas DataFrame with landmark data and optional label column
        label_column: Name of the column containing labels. Auto-detected if None.
    
    Returns:
        Tuple of (samples, labels) where samples is a list of numpy arrays
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a Pandas DataFrame")
    
    if df.empty:
        raise ValueError("DataFrame is empty")
    
    if label_column is None:
        label_column = _detect_label_column(df)
    
    feature_cols = [c for c in df.columns if c != label_column]
    
    if not feature_cols:
        raise ValueError("No feature columns found in DataFrame")
    
    samples = []
    labels = []
    
    for idx in range(len(df)):
        row_data = df[feature_cols].iloc[idx]
        
        if row_data.dtype == object and len(feature_cols) == 1:
            try:
                val = row_data.iloc[0]
                if isinstance(val, str):
                    sample = np.array(ast.literal_eval(val), dtype=np.float32)
                elif isinstance(val, (list, np.ndarray)):
                    sample = np.array(val, dtype=np.float32)
                else:
                    sample = np.array([float(val)], dtype=np.float32)
            except:
                try:
                    sample = np.array(row_data.values.tolist(), dtype=np.float32)
                except:
                    continue
        else:
            try:
                sample = np.array(row_data.values.tolist(), dtype=np.float32)
            except:
                continue
        
        if sample.ndim == 1 and len(sample) % INPUT_DIM == 0 and len(sample) > 0:
            num_frames = len(sample) // INPUT_DIM
            sample = sample.reshape(num_frames, INPUT_DIM)
        
        samples.append(sample)
        
        if label_column:
            labels.append(str(df[label_column].iloc[idx]))
        else:
            labels.append('unknown')
    
    if not samples:
        raise ValueError("Could not extract any valid samples from DataFrame")
    
    return samples, labels


def _load_json_dataset(file_path):
    """Load dataset from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return _parse_json_list(data)
    elif isinstance(data, dict):
        return _parse_json_dict(data)
    
    return None, None


def _parse_json_list(data):
    """Parse JSON list format: [{"landmarks": [...], "label": "class"}, ...]"""
    samples = []
    labels = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if 'landmarks' in item:
            samples.append(np.array(item['landmarks'], dtype=np.float32))
        elif 'data' in item:
            samples.append(np.array(item['data'], dtype=np.float32))
        else:
            continue
        labels.append(item.get('label', item.get('class', 'unknown')))
    return samples, labels


def _parse_json_dict(data):
    """Parse JSON dict format: {"data": [...], "labels": [...]}"""
    if 'data' in data and 'labels' in data:
        return [np.array(x, dtype=np.float32) for x in data['data']], data['labels']
    elif 'landmarks' in data:
        return [np.array(x, dtype=np.float32) for x in data['landmarks']], data.get('labels', [])
    return None, None


def _load_tabular_dataset(file_path, ext):
    """Load dataset from CSV or Excel file."""
    if ext == '.csv':
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)
    
    return load_dataset_from_dataframe(df)


def _load_npy_dataset(file_path):
    """Load dataset from NumPy file."""
    data = np.load(file_path)
    if data.ndim == 2:
        return [data[i] for i in range(len(data))], ['unknown'] * len(data)
    elif data.ndim == 3:
        return [data[i] for i in range(len(data))], ['unknown'] * len(data)
    else:
        return [data], ['unknown']


def _detect_label_column(df):
    """Auto-detect the label column in a DataFrame."""
    for col in ['label', 'class', 'target', 'y', 'sign', 'gesture', 'category']:
        if col in df.columns:
            return col
    
    for col in df.columns:
        if df[col].dtype == object:
            unique_count = df[col].nunique()
            if 1 < unique_count < len(df) * 0.5:
                return col
    
    return None


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
                
                self.frame_ready.emit(frame.copy(), hand_features.copy())
                
        except Exception as e:
            self.error_occurred.emit(f"Camera error: {str(e)}")
        finally:
            if self._holistic:
                self._holistic.close()
        
    def _extract_hands(self, results):
        features = np.zeros(INPUT_DIM, dtype=np.float32)
        
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


class TrainingWorker(QThread):
    """Background thread for model training."""
    epoch_done = pyqtSignal(int, int, float, float)
    training_complete = pyqtSignal(float, str)
    training_error = pyqtSignal(str)
    progress_update = pyqtSignal(str)
    
    def __init__(self, model, train_loader, val_loader, device, epochs, class_labels):
        super().__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.class_labels = class_labels
        self.running = True
        
    def run(self):
        try:
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
            
            best_val_acc = 0
            best_state = None
            
            Path('checkpoints').mkdir(exist_ok=True)
            
            for epoch in range(self.epochs):
                if not self.running:
                    break
                    
                self.model.train()
                train_loss = 0
                train_correct = 0
                train_total = 0
                
                for batch_x, batch_y in self.train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += batch_y.size(0)
                    train_correct += predicted.eq(batch_y).sum().item()
                
                train_acc = 100. * train_correct / train_total if train_total > 0 else 0
                
                self.model.eval()
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_x, batch_y in self.val_loader:
                        batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_x)
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()
                
                val_acc = 100. * val_correct / val_total if val_total > 0 else 0
                
                scheduler.step()
                
                self.epoch_done.emit(epoch + 1, self.epochs, train_acc, val_acc)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            
            if best_state:
                torch.save({
                    'epoch': self.epochs - 1,
                    'model_state_dict': best_state,
                    'val_acc': best_val_acc,
                    'class_labels': self.class_labels
                }, 'checkpoints/best_model.pt')
            
            self.training_complete.emit(best_val_acc, 'checkpoints/best_model.pt')
            
        except Exception as e:
            self.training_error.emit(str(e))
    
    def cancel(self):
        self.running = False


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator")
        self.setMinimumSize(1000, 750)
        
        self.model = None
        self.class_labels = []
        self.dataset_samples = []
        self.dataset_labels = []
        self.dataset_df = None
        self.frame_buffer = deque(maxlen=DEFAULT_NUM_FRAMES)
        self.translator_thread = None
        self.training_worker = None
        self.current_prediction = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        
        self.setup_ui()
        self.load_settings()
        
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        
        self.tabs = QTabWidget()
        
        self.tabs.addTab(self.create_translate_tab(), "Translate")
        self.tabs.addTab(self.create_dataset_tab(), "Dataset")
        
        layout.addWidget(self.tabs)
        
    def create_translate_tab(self):
        widget = QWidget()
        layout = QHBoxLayout()
        widget.setLayout(layout)
        
        left_panel = self.create_control_panel()
        layout.addWidget(left_panel, 1)
        
        right_panel = self.create_video_panel()
        layout.addWidget(right_panel, 2)
        
        return widget
        
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
        self.num_frames_spin.setValue(DEFAULT_NUM_FRAMES)
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
        self.log_text.setMaximumHeight(120)
        log_layout.addWidget(self.log_text)
        
        log_group.setLayout(log_layout)
        layout.addWidget(log_group)
        
        group.setLayout(layout)
        return group
        
    def create_dataset_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        load_group = QGroupBox("Load Dataset")
        load_layout = QVBoxLayout()
        
        format_label = QLabel("Supported formats: JSON, CSV, Excel (.xlsx), NumPy (.npy), Pandas DataFrame")
        format_label.setStyleSheet("color: #666;")
        load_layout.addWidget(format_label)
        
        btn_row = QHBoxLayout()
        
        self.btn_load_dataset = QPushButton("Load Dataset File")
        self.btn_load_dataset.clicked.connect(self.load_dataset_file)
        btn_row.addWidget(self.btn_load_dataset)
        
        self.btn_load_dataframe = QPushButton("Load from DataFrame")
        self.btn_load_dataframe.clicked.connect(self.load_dataframe_dialog)
        btn_row.addWidget(self.btn_load_dataframe)
        
        load_layout.addLayout(btn_row)
        
        self.dataset_info_label = QLabel("No dataset loaded")
        self.dataset_info_label.setStyleSheet("color: #888;")
        load_layout.addWidget(self.dataset_info_label)
        
        load_group.setLayout(load_layout)
        layout.addWidget(load_group)
        
        preview_group = QGroupBox("Dataset Preview")
        preview_layout = QVBoxLayout()
        
        self.dataset_table = QTableWidget()
        self.dataset_table.setMaximumHeight(200)
        preview_layout.addWidget(self.dataset_table)
        
        preview_group.setLayout(preview_layout)
        layout.addWidget(preview_group)
        
        train_group = QGroupBox("Training")
        train_layout = QFormLayout()
        
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 500)
        self.epochs_spin.setValue(10)
        train_layout.addRow("Epochs:", self.epochs_spin)
        
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        train_layout.addRow("Batch Size:", self.batch_size_spin)
        
        btn_train_row = QHBoxLayout()
        
        self.btn_train = QPushButton("Train Model")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_train.setEnabled(False)
        btn_train_row.addWidget(self.btn_train)
        
        self.btn_cancel_train = QPushButton("Cancel")
        self.btn_cancel_train.clicked.connect(self.cancel_training)
        self.btn_cancel_train.setEnabled(False)
        btn_train_row.addWidget(self.btn_cancel_train)
        
        train_layout.addRow("", btn_train_row)
        
        self.training_log_text = QTextEdit()
        self.training_log_text.setReadOnly(True)
        self.training_log_text.setMaximumHeight(150)
        train_layout.addRow("Training Log:", self.training_log_text)
        
        train_group.setLayout(train_layout)
        layout.addWidget(train_group)
        
        layout.addStretch()
        
        return widget
        
    def log(self, message):
        self.log_text.append(message)
        
    def log_training(self, message):
        self.training_log_text.append(message)
        
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
            
        try:
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)
            
            num_classes = len(self.class_labels) if self.class_labels else 100
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict):
                state_dict = checkpoint
            else:
                raise ValueError("Invalid checkpoint format")
            
            for key, param in list(state_dict.items())[:10]:
                if len(param.shape) == 4 and param.shape[1] in [1, 3]:
                    raise ValueError("This model appears to use image input (Conv2D with 1/3 input channels), but this app requires a landmark-based model.")
            
            self.model = create_model(num_classes=num_classes)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            self.model_path = path
            self.model_loaded = True
            self.model_label.setText(f"Model: {Path(path).name}")
            
            if self.class_labels:
                self.btn_start.setEnabled(True)
                
            self.log(f"Loaded model from {Path(path).name}")
                
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
            
            if self.model_loaded:
                self.btn_start.setEnabled(True)
                    
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load labels: {e}")
    
    def load_dataset_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Dataset File", "", 
                                              "Dataset Files (*.json *.csv *.xlsx *.xls *.npy);;All Files (*)")
        if not path:
            return
        
        try:
            samples, labels = load_dataset(path)
            
            if samples is None or len(samples) == 0:
                raise ValueError("Could not parse dataset file or dataset is empty")
            
            self._apply_dataset(samples, labels)
            self.log(f"Loaded dataset from file: {Path(path).name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load dataset:\n{str(e)}")
            self.log(f"Dataset error: {e}")
    
    def load_dataframe_dialog(self):
        """Show dialog for loading DataFrame from file."""
        if not PANDAS_AVAILABLE:
            QMessageBox.warning(self, "Missing Dependency", "Pandas is required.\nRun: pip install pandas openpyxl")
            return
            
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV/Excel File", "", 
                                              "Tabular Files (*.csv *.xlsx *.xls);;All Files (*)")
        if not path:
            return
        
        try:
            ext = Path(path).suffix.lower()
            if ext == '.csv':
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            
            samples, labels = load_dataset_from_dataframe(df)
            
            if not samples:
                raise ValueError("No valid samples found in DataFrame")
            
            self.dataset_df = df
            self._apply_dataset(samples, labels)
            self.log(f"Loaded DataFrame from: {Path(path).name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load DataFrame:\n{str(e)}")
            self.log(f"DataFrame error: {e}")
    
    def load_dataset_from_dataframe(self, df, label_column=None):
        """Load dataset directly from a Pandas DataFrame.
        
        This method can be called programmatically or via the UI dialog.
        
        Args:
            df: Pandas DataFrame with landmark data
            label_column: Name of the label column (auto-detected if None)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a Pandas DataFrame")
        
        samples, labels = load_dataset_from_dataframe(df, label_column)
        
        if not samples:
            raise ValueError("No valid samples found in DataFrame")
        
        self.dataset_df = df
        self._apply_dataset(samples, labels)
        self.log(f"Loaded DataFrame with {len(samples)} samples, {len(self.class_labels)} classes")
        
        return samples, labels
    
    def _apply_dataset(self, samples, labels):
        """Apply loaded dataset to the app state."""
        self.dataset_samples = samples
        self.dataset_labels = labels
        
        unique_labels = sorted(set(labels))
        self.class_labels = unique_labels
        
        self.dataset_info_label.setText(f"Loaded {len(samples)} samples, {len(unique_labels)} classes")
        self.log(f"Dataset: {len(samples)} samples, {len(unique_labels)} classes")
        
        preview_count = min(10, len(samples))
        self.dataset_table.setRowCount(preview_count)
        self.dataset_table.setColumnCount(2)
        self.dataset_table.setHorizontalHeaderLabels(["Sample Shape", "Label"])
        
        for i in range(preview_count):
            shape_str = str(samples[i].shape) if hasattr(samples[i], 'shape') else str(type(samples[i]))
            self.dataset_table.setItem(i, 0, QTableWidgetItem(shape_str))
            self.dataset_table.setItem(i, 1, QTableWidgetItem(str(labels[i])))
        
        self.dataset_table.resizeColumnsToContents()
        
        self.btn_train.setEnabled(True)
        
    def start_training(self):
        if not self.dataset_samples:
            QMessageBox.warning(self, "Error", "No dataset loaded")
            return
        
        if not self.class_labels:
            QMessageBox.warning(self, "Error", "No classes found in dataset")
            return
        
        if len(self.dataset_samples) < 2:
            QMessageBox.warning(self, "Error", "Need at least 2 samples to train")
            return
        
        self.btn_train.setEnabled(False)
        self.btn_cancel_train.setEnabled(True)
        self.training_log_text.clear()
        self.log_training("Starting training...")
        
        try:
            num_classes = len(self.class_labels)
            model = create_model(num_classes=num_classes)
            model.to(self.device)
            
            label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
            labels = [label_to_idx[l] for l in self.dataset_labels]
            
            sequences = []
            for sample in self.dataset_samples:
                if sample.ndim == 1:
                    num_frames = len(sample) // INPUT_DIM
                    seq = sample.reshape(num_frames, INPUT_DIM)
                else:
                    seq = sample
                
                if seq.shape[0] < DEFAULT_NUM_FRAMES:
                    padding = np.zeros((DEFAULT_NUM_FRAMES - seq.shape[0], INPUT_DIM), dtype=np.float32)
                    seq = np.vstack([seq, padding])
                else:
                    seq = seq[:DEFAULT_NUM_FRAMES]
                
                sequences.append(seq)
            
            sequences = np.array(sequences, dtype=np.float32)
            labels = np.array(labels, dtype=np.int64)
            
            dataset_size = len(sequences)
            train_size = max(1, int(0.8 * dataset_size))
            indices = np.random.permutation(dataset_size)
            
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            
            if len(val_indices) == 0:
                val_indices = train_indices[-1:]
                train_indices = train_indices[:-1]
            
            train_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(sequences[train_indices]),
                torch.LongTensor(labels[train_indices])
            )
            val_dataset = torch.utils.data.TensorDataset(
                torch.FloatTensor(sequences[val_indices]),
                torch.LongTensor(labels[val_indices])
            )
            
            actual_batch_size = min(self.batch_size_spin.value(), len(train_dataset))
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=actual_batch_size)
            
            self.training_worker = TrainingWorker(
                model, train_loader, val_loader, self.device,
                self.epochs_spin.value(), self.class_labels
            )
            
            self.training_worker.epoch_done.connect(self.on_epoch_done)
            self.training_worker.training_complete.connect(self.on_training_complete)
            self.training_worker.training_error.connect(self.on_training_error)
            
            self.training_worker.start()
            
        except Exception as e:
            self.log_training(f"Error: {str(e)}")
            self.log_training(traceback.format_exc())
            QMessageBox.warning(self, "Training Error", str(e))
            self.btn_train.setEnabled(True)
            self.btn_cancel_train.setEnabled(False)
    
    def cancel_training(self):
        if self.training_worker:
            self.training_worker.cancel()
            self.log_training("Training cancelled")
            self.btn_train.setEnabled(True)
            self.btn_cancel_train.setEnabled(False)
    
    def on_epoch_done(self, epoch, total_epochs, train_acc, val_acc):
        self.log_training(f"Epoch {epoch}/{total_epochs} - Train: {train_acc:.1f}% Val: {val_acc:.1f}%")
    
    def on_training_complete(self, best_val_acc, save_path):
        self.log_training(f"Training complete! Best val accuracy: {best_val_acc:.1f}%")
        
        try:
            checkpoint = torch.load(save_path, map_location=self.device, weights_only=False)
            self.model = create_model(num_classes=len(self.class_labels))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.model_loaded = True
            self.model_label.setText("Model: Trained")
            self.btn_start.setEnabled(True)
            
            self.log("Model trained and ready!")
            
            QMessageBox.information(self, "Training Complete", 
                                   f"Model trained with {best_val_acc:.1f}% validation accuracy.\nSaved to {save_path}")
        except Exception as e:
            self.log(f"Error loading trained model: {e}")
        
        self.btn_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self.training_worker = None
    
    def on_training_error(self, error_msg):
        self.log_training(f"Training error: {error_msg}")
        QMessageBox.warning(self, "Training Error", error_msg)
        self.btn_train.setEnabled(True)
        self.btn_cancel_train.setEnabled(False)
        self.training_worker = None
                
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
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        
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
                padding = np.zeros((self.num_frames_spin.value() - sequence.shape[0], INPUT_DIM), dtype=np.float32)
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
                    self.num_frames_spin.setValue(settings.get('num_frames', DEFAULT_NUM_FRAMES))
                    self.confidence_spin.setValue(settings.get('confidence', 0.7))
                    self.camera_spin.setValue(settings.get('camera_index', 0))
            except:
                pass
                
    def closeEvent(self, event):
        if self.translator_thread:
            self.translator_thread.stop()
            self.translator_thread = None
            
        if self.training_worker:
            self.training_worker.cancel()
            self.training_worker.wait(2000)
            
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
    
    if not PANDAS_AVAILABLE:
        print("NOTE: Pandas not installed. Dataset loading from CSV/Excel will be disabled.")
        print("Run: pip install pandas openpyxl")
        
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()