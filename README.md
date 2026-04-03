# Sign Language Translator

A camera-based ML application that translates sign language into English text using pose estimation and deep learning.

## Desktop Application

The desktop app provides a GUI for real-time sign language translation:

```bash
pip install -r requirements.txt
python app.py
```

### Features
- Load trained model and class labels via GUI
- Real-time webcam feed with pose detection
- Adjustable sequence length and confidence threshold
- Logging panel for debugging

### Requirements
- Windows or Linux
- Webcam
- Trained model (`.pt`/`.pth` file)
- Class labels (`.json` file)

---

## Architecture (ML Pipeline)

Based on the ML pipeline from the report:
1. **Input Capture** - Webcam/camera with frame normalization
2. **Pose Estimation** - MediaPipe Holistic for hand/body landmarks (21 hand + 33 pose + 468 face keypoints)
3. **Temporal Modeling** - Transformer, CNN, LSTM, or Hybrid models
4. **Sign Recognition** - Classification to gloss labels
5. **Translation** - Sequence-to-text output

---

## Project Structure

| File | Description |
|------|-------------|
| `app.py` | Desktop GUI application |
| `data_loader.py` | Dataset loaders with easy customization |
| `preprocessing.py` | Video preprocessing & landmark extraction |
| `models.py` | Multiple model architectures (Transformer, CNN, LSTM, Hybrid) |
| `training.py` | Training pipeline with validation |
| `realtime_translator.py` | Real-time webcam inference (CLI) |
| `dataset_preparation.py` | Tools to convert videos to landmarks |
| `example_usage.py` | Quick start examples |
| `requirements.txt` | Dependencies |

---

## Quick Start (CLI)

### 1. Prepare Your Dataset

Option A: Convert existing video folder to landmarks
```bash
python dataset_preparation.py convert-folder -i /path/to/videos -o data/my_dataset
```

Option B: Capture new samples from webcam
```bash
python dataset_preparation.py capture -o data/my_dataset -n 20 --class-name "hello"
```

Option C: Place your data manually:
```
data/
  my_dataset/
    hello/
      sample_001.npy
    thanks/
      sample_001.npy
    class_labels.json
```

### 2. Train Model

```bash
python example_usage.py train
```

### 3. Run Real-time Translation

```python
from realtime_translator import create_translator

translator = create_translator(
    model_path='checkpoints/best_model.pt',
    class_labels_path='data/my_dataset/class_labels.json'
)
translator.start()
```

---

## Adding Your Own Dataset

The dataset system is designed for easy customization:

### Format 1: Folder Structure (Recommended)
```
data/
  your_dataset/
    class1/
      sample1.npy  # shape: (30, 126)
    class2/
      sample1.npy
```

### Format 2: CSV-based
```python
from data_loader import CustomDataset

dataset = CustomDataset.create_from_csv('annotations.csv', 'landmarks/')
```

### Custom Dataset Class

Extend the base class for specialized formats:

```python
from data_loader import SignLanguageDataset

class MyDataset(SignLanguageDataset):
    def _load_samples(self):
        # Custom loading logic
        return samples
```

---

## Model Options

Available in `models.py`:

| Model | Description | Best For |
|-------|-------------|----------|
| `transformer` | Self-attention over sequences | Long-range dependencies |
| `cnn1d` | 1D convolutional layers | Fast training, local patterns |
| `lstm` | Bidirectional LSTM | Sequential dependencies |
| `hybrid` | CNN + Transformer | Both local and global features |

---

## Configuration

Key parameters:

- **Frames per sample**: Default 30 (edit in `data_loader.py`)
- **Input dimension**: 126 (21 hand landmarks × 3 coords × 2 hands)
- **Model hyperparameters**: Edit in `models.py` constructor calls

---

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for training)