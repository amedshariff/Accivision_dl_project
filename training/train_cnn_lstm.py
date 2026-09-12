"""
Train CNN + LSTM Model Script (Person 3 - Level 5)
Trains the temporal classifier on video sequences and saves the best model checkpoint.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.optim as optim
from models.cnn_lstm_model import CNNLSTMModel
from training.dataset_loader import get_data_loader
from training.loss_functions import WeightedAccidentLoss

DEFAULT_CHECKPOINT_PATH = Path("models/saved_models/best_model.pth")

def train_model(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/validation.csv",
    epochs: int = 5,
    lr: float = 1e-4,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)
):
    print("=" * 60)
    print("TRAINING CNN + LSTM ACCIDENT DETECTION MODEL")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    train_loader = get_data_loader(train_csv, batch_size=2, shuffle=True)
    val_loader = get_data_loader(val_csv, batch_size=2, shuffle=False)

    model = CNNLSTMModel(pretrained_cnn=True, freeze_cnn=True).to(device)
    criterion = WeightedAccidentLoss(accident_weight=2.0)
    optimizer = optim.Adam(model.lstm.parameters(), lr=lr)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for sequences, labels, _ in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * sequences.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for sequences, labels, _ in val_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                logits = model(sequences)
                loss = criterion(logits, labels)
                val_loss += loss.item() * sequences.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  [+] Saved new best model checkpoint to {checkpoint_path}")

    print("=" * 60)
    print(f"Training completed! Best checkpoint: {checkpoint_path}")
    return model

if __name__ == "__main__":
    train_model(epochs=3)
