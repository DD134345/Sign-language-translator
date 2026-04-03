import torch
import numpy as np
import json
from pathlib import Path
from data_loader import SignLanguageDataset, get_data_loader
from models import get_model
from training import Trainer


def prepare_example_dataset():
    """Create an example dataset structure for demonstration."""
    data_dir = Path('data/my_dataset')
    data_dir.mkdir(exist_ok=True, parents=True)
    
    class_names = ['hello', 'thanks', 'yes', 'no', 'please']
    
    for class_name in class_names:
        class_dir = data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        for i in range(5):
            sequence = np.random.randn(30, 126).astype(np.float32)
            np.save(class_dir / f'sample_{i}.npy', sequence)
    
    with open(data_dir / 'class_labels.json', 'w') as f:
        json.dump(class_names, f, indent=2)
    
    print(f"Created example dataset at {data_dir}")
    print(f"Classes: {class_names}")


def train_example():
    """Train a model with the example dataset."""
    data_dir = 'data/my_dataset'
    num_classes = 5
    batch_size = 4
    
    dataset = SignLanguageDataset(data_dir)
    print(f"Loaded {len(dataset)} samples")
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = get_data_loader(train_dataset, batch_size=batch_size)
    val_loader = get_data_loader(val_dataset, batch_size=batch_size)
    
    model = get_model(
        model_type='transformer',
        num_classes=num_classes,
        input_dim=126
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_epochs=10
    )
    
    print("Starting training...")
    trainer.train()
    print("Training complete!")


def quick_test():
    """Quick test to verify everything works."""
    print("Testing imports...")
    from data_loader import SignLanguageDataset
    from models import get_model
    from preprocessing import VideoPreprocessor
    from training import Trainer
    print("All imports successful!")
    
    print("\nTesting model creation...")
    model = get_model('transformer', num_classes=10, input_dim=126)
    dummy_input = torch.randn(2, 30, 126)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    
    print("\nTesting preprocessing...")
    preprocessor = VideoPreprocessor()
    print("Preprocessor initialized successfully")
    preprocessor.close()
    
    print("\nAll tests passed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python example_usage.py [prepare|train|test]")
        sys.exit(1)
    
    mode = sys.argv[1]
    
    if mode == 'prepare':
        prepare_example_dataset()
    elif mode == 'train':
        train_example()
    elif mode == 'test':
        quick_test()
    else:
        print(f"Unknown mode: {mode}")