import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

class SignLanguageDataset(Dataset):
    """Base dataset class - extend this for your custom dataset."""
    
    def __init__(self, data_dir, num_frames=30, transform=None):
        self.data_dir = Path(data_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        """Load samples from dataset. Override this for custom datasets."""
        samples = []
        classes = sorted([d.name for d in self.data_dir.iterdir() if d.is_dir()])
        
        for class_idx, class_name in enumerate(classes):
            class_dir = self.data_dir / class_name
            for video_file in class_dir.glob('*.npy'):
                samples.append({
                    'path': str(video_file),
                    'label': class_idx,
                    'class_name': class_name
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        landmarks = np.load(sample['path'])
        
        if self.transform:
            landmarks = self.transform(landmarks)
            
        return torch.FloatTensor(landmarks), sample['label']


class VideoFrameDataset(Dataset):
    """Dataset that loads video frames directly (for video-based datasets)."""
    
    def __init__(self, video_dir, landmarks_dir, num_frames=30, transform=None):
        self.video_dir = Path(video_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.num_frames = num_frames
        self.transform = transform
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        video_files = list(self.video_dir.glob('*.mp4')) + list(self.video_dir.glob('*.avi'))
        
        for video_file in video_files:
            video_name = video_file.stem
            landmarks_file = self.landmarks_dir / f"{video_name}.npy"
            
            if landmarks_file.exists():
                samples.append({
                    'video_path': str(video_file),
                    'landmarks_path': str(landmarks_file),
                    'label': 0
                })
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        landmarks = np.load(sample['landmarks_path'])
        
        return torch.FloatTensor(landmarks), sample['label']


class CustomDataset:
    """Template for creating your own dataset loader."""
    
    @staticmethod
    def create_from_csv(csv_path, landmarks_dir, num_frames=30):
        """
        Create dataset from CSV file with columns:
        [path, label] or [video_path, landmarks_path, label]
        """
        import pandas as pd
        df = pd.read_csv(csv_path)
        # Implement based on your CSV format
        pass
    
    @staticmethod
    def create_from_folder(base_dir, num_frames=30):
        """
        Create dataset from folder structure:
        base_dir/
            class1/
                sample1.npy
                sample2.npy
            class2/
                sample1.npy
        """
        return SignLanguageDataset(base_dir, num_frames)


def get_data_loader(dataset, batch_size=8, shuffle=True, num_workers=4):
    """Create a DataLoader with sensible defaults."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )