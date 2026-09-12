"""
End-to-End CNN + LSTM Accident Detection Model (Person 3 - Level 5)
Accepts video tensor (B, T, C, H, W) -> CNN Backbone -> LSTM Classifier -> Accident Probability
"""
import torch
import torch.nn as nn
from models.cnn_model import CNNFeatureExtractor
from models.lstm_model import LSTMClassifier

class CNNLSTMModel(nn.Module):
    def __init__(
        self,
        pretrained_cnn: bool = True,
        freeze_cnn: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3
    ):
        super(CNNLSTMModel, self).__init__()
        self.cnn = CNNFeatureExtractor(pretrained=pretrained_cnn, freeze_backbone=freeze_cnn)
        self.lstm = LSTMClassifier(
            input_dim=self.cnn.feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=1,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, T, C, H, W)
        Output: (B, 1) logit
        """
        B, T, C, H, W = x.shape
        # Flatten batch and time dimensions for CNN pass
        x_reshaped = x.view(B * T, C, H, W)
        features = self.cnn(x_reshaped)  # (B * T, 512)
        
        # Reshape back to temporal sequence for LSTM
        features_seq = features.view(B, T, -1)  # (B, T, 512)
        logits = self.lstm(features_seq)        # (B, 1)
        return logits

    def predict_probability(self, x: torch.Tensor) -> float:
        """
        Inference helper returning accident probability in [0.0, 1.0].
        """
        self.eval()
        with torch.no_grad():
            if x.dim() == 4:  # (T, C, H, W)
                x = x.unsqueeze(0)  # (1, T, C, H, W)
            logits = self.forward(x)
            prob = torch.sigmoid(logits).item()
        return round(prob, 4)
