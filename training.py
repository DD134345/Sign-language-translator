import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path


class Trainer:
    """Training pipeline for sign language recognition."""
    
    def __init__(self, model, train_loader, val_loader, device='cuda', 
                 learning_rate=1e-4, num_epochs=50, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs, eta_min=1e-6
        )
        
        self.num_epochs = num_epochs
        self.best_val_acc = 0.0
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(tqdm(self.train_loader, desc='Training')):
            data, labels = data.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(data)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(self.train_loader), 100. * correct / total
    
    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in tqdm(self.val_loader, desc='Validating'):
                data, labels = data.to(self.device), labels.to(self.device)
                
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return running_loss / len(self.val_loader), 100. * correct / total
    
    def train(self):
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            
            self.scheduler.step()
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc)
    
    def save_checkpoint(self, epoch, val_acc):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
        }
        torch.save(checkpoint, self.checkpoint_dir / 'best_model.pt')
        print(f"Checkpoint saved: {self.checkpoint_dir / 'best_model.pt'}")


class CTCTrainer:
    """Trainer for CTC-based sign segmentation."""
    
    def __init__(self, model, train_loader, device='cuda', learning_rate=1e-4):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.ctc_loss = nn.CTCLLoss(blank=0, zero_infinity=True)
        
    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0
            
            for batch in tqdm(self.train_loader, desc=f'Epoch {epoch+1}'):
                data, targets, input_lengths, target_lengths = batch
                data = data.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(data)
                
                output_lengths = torch.full((data.size(0),), outputs.size(1), dtype=torch.long)
                
                loss = self.ctc_loss(outputs.log_softmax(2), targets, output_lengths, target_lengths)
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"Epoch {epoch+1} - Loss: {total_loss/len(self.train_loader):.4f}")