"""
LSTM Temporal Classifier Module (Person 3 - Level 5)
Processes sequential feature representations over time to detect accident dynamics.
"""
import torch
import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 1,
        dropout: float = 0.3
    ):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, input_dim)
        Output: (B, 1) logit for binary accident classification
        """
        lstm_out, (hn, cn) = self.lstm(x)  # lstm_out: (B, T, hidden_dim)
        # Use last temporal output or pooled representation
        last_temporal = lstm_out[:, -1, :]  # (B, hidden_dim)
        logits = self.classifier(last_temporal)  # (B, 1)
        return logits
