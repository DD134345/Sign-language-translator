import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpatialGraphConv(nn.Module):
    """Graph convolution for body structure."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x, adj):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)


class TemporalTransformer(nn.Module):
    """Transformer for temporal modeling of sign sequences."""
    
    def __init__(self, 
                 input_dim: int = 126,  # 21 hand landmarks * 3 coords * 2 hands
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 num_classes: int = 100):
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x, mask=None):
        # x: (batch, seq_len, input_dim)
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"Expected input_dim {self.input_dim}, got {x.shape[-1]}")
            
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Use last frame or pooled representation
        pooled = x.mean(dim=1)
        
        return self.classifier(pooled)


class Conv1DBlock(nn.Module):
    """1D Convolutional block for temporal features."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        
    def forward(self, x):
        return self.pool(self.relu(self.bn(self.conv(x))))


class CNN1DModel(nn.Module):
    """1D CNN for spatiotemporal pattern capture."""
    
    def __init__(self, 
                 input_dim: int = 126,
                 num_classes: int = 100,
                 hidden_dims: list = [128, 256, 512]):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for out_dim in hidden_dims:
            layers.append(Conv1DBlock(in_dim, out_dim))
            in_dim = out_dim
        
        self.conv_layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        
        pooled = x.mean(dim=2)
        return self.classifier(pooled)


class LSTMModel(nn.Module):
    """LSTM for sequential modeling."""
    
    def __init__(self, 
                 input_dim: int = 126,
                 hidden_dim: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.3,
                 num_classes: int = 100):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        
        # Concatenate forward and backward final states
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        return self.classifier(hidden)


class HybridCNNTransformer(nn.Module):
    """Hybrid model combining CNN local patterns with Transformer global context."""
    
    def __init__(self,
                 input_dim: int = 126,
                 d_model: int = 256,
                 nhead: int = 8,
                 cnn_channels: list = [64, 128],
                 num_layers: int = 3,
                 num_classes: int = 100):
        super().__init__()
        
        # CNN feature extraction
        cnn_layers = []
        in_ch = input_dim
        for out_ch in cnn_channels:
            cnn_layers.append(Conv1DBlock(in_ch, out_ch))
            in_ch = out_ch
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # Project to transformer dimension
        self.proj = nn.Linear(cnn_channels[-1], d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # CNN: (batch, seq, features) -> (batch, seq, cnn_out)
        x = x.transpose(1, 2)
        x = self.cnn(x)
        x = x.transpose(1, 2)
        
        x = self.proj(x)
        x = self.pos_encoder(x)
        
        x = self.transformer(x)
        
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class SignRecognizer(nn.Module):
    """Sign recognition model that supports multiple architectures."""
    
    def __init__(self, 
                 model_type: str = 'transformer',
                 input_dim: int = 126,
                 num_classes: int = 100,
                 **kwargs):
        super().__init__()
        
        self.model_type = model_type
        
        if model_type == 'transformer':
            self.model = TemporalTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == 'cnn1d':
            self.model = CNN1DModel(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == 'lstm':
            self.model = LSTMModel(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs
            )
        elif model_type == 'hybrid':
            self.model = HybridCNNTransformer(
                input_dim=input_dim,
                num_classes=num_classes,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def forward(self, x):
        return self.model(x)


def get_model(model_type: str = 'transformer', 
              num_classes: int = 100,
              input_dim: int = 126):
    """Factory function to create models."""
    return SignRecognizer(model_type=model_type, num_classes=num_classes, input_dim=input_dim)