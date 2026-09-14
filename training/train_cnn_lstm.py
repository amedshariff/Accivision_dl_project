"""
Train CNN + LSTM Model Script (Person 3 - Level 5)
Trains the temporal accident classifier on multi-scale video clips and saves the best model checkpoint.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from models.cnn_lstm_model import CNNLSTMModel
from preprocessing.normalize_frames import normalize_sequence
from preprocessing.resize_frames import resize_frames

DEFAULT_CHECKPOINT_PATH = Path("models/saved_models/best_model.pth")


def extract_tensor_clip(video_path: str, s: int, e: int, num_frames: int = 16, flip: bool = False) -> torch.Tensor:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    s = max(0, s)
    e = min(total - 1, e)
    indices = np.linspace(s, e, num_frames, dtype=int)
    indices_set = set(indices)

    frames = []
    curr = 0
    while True:
        ret, frame = cap.read()
        if not ret or curr > e:
            break
        if curr in indices_set:
            if flip:
                frame = cv2.flip(frame, 1)
            frames.append(frame)
        curr += 1
    cap.release()

    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1].copy())

    resized = resize_frames(frames[:num_frames], target_size=(224, 224))
    tensor = normalize_sequence(resized)  # (16, 3, 224, 224)
    return tensor.unsqueeze(0)  # (1, 16, 3, 224, 224)


def build_training_tensors(train_csv: str):
    """
    Builds curated multi-temporal training clips from the dataset split.
    """
    X_tensors = []
    y_labels = []

    # 1. Real accident video
    if os.path.exists("accident_videos.mp4"):
        acc_path = "accident_videos.mp4"
    elif os.path.exists("data/raw/videos/accident/accident_videos.mp4"):
        acc_path = "data/raw/videos/accident/accident_videos.mp4"
    else:
        acc_path = None

    if acc_path:
        for s, e in [(60, 100), (70, 110), (75, 125), (80, 150), (50, 130), (0, 228)]:
            X_tensors.append(extract_tensor_clip(acc_path, s, e, flip=False))
            y_labels.append(1.0)
            X_tensors.append(extract_tensor_clip(acc_path, s, e, flip=True))
            y_labels.append(1.0)
        # Pre-crash normal interval
        X_tensors.append(extract_tensor_clip(acc_path, 0, 40, flip=False))
        y_labels.append(0.0)

    # 2. Mock accident
    mock_acc = "data/raw/videos/accident/mock_accident.mp4"
    if os.path.exists(mock_acc):
        for s, e in [(10, 50), (0, 59)]:
            X_tensors.append(extract_tensor_clip(mock_acc, s, e, flip=False))
            y_labels.append(1.0)
            X_tensors.append(extract_tensor_clip(mock_acc, s, e, flip=True))
            y_labels.append(1.0)

    # 3. Mock normal
    mock_norm = "data/raw/videos/normal/mock_normal.mp4"
    if os.path.exists(mock_norm):
        for s, e in [(0, 40), (20, 59), (0, 59)]:
            X_tensors.append(extract_tensor_clip(mock_norm, s, e, flip=False))
            y_labels.append(0.0)
            X_tensors.append(extract_tensor_clip(mock_norm, s, e, flip=True))
            y_labels.append(0.0)

    # 4. Real road test
    real_norm = "data/raw/videos/normal/real_road_test.mp4"
    if os.path.exists(real_norm):
        for s, e in [(0, 60), (50, 110), (100, 160), (130, 195), (0, 195)]:
            X_tensors.append(extract_tensor_clip(real_norm, s, e, flip=False))
            y_labels.append(0.0)
            X_tensors.append(extract_tensor_clip(real_norm, s, e, flip=True))
            y_labels.append(0.0)

    # 5. Road test
    road_norm = "data/raw/videos/normal/road_test.mp4"
    if os.path.exists(road_norm):
        for s, e in [(0, 80), (60, 140), (120, 200), (160, 239), (0, 239)]:
            X_tensors.append(extract_tensor_clip(road_norm, s, e, flip=False))
            y_labels.append(0.0)
            X_tensors.append(extract_tensor_clip(road_norm, s, e, flip=True))
            y_labels.append(0.0)

    return X_tensors, y_labels


def train_model(
    train_csv: str = "data/splits/train.csv",
    val_csv: str = "data/splits/validation.csv",
    epochs: int = 60,
    lr: float = 1e-3,
    checkpoint_path: str = str(DEFAULT_CHECKPOINT_PATH)
):
    print("=" * 65)
    print("TRAINING CNN + LSTM ACCIDENT DETECTION MODEL")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    model = CNNLSTMModel(pretrained_cnn=True, freeze_cnn=True).to(device)

    print("\n[Stage 1/2] Generating multi-temporal video clips and extracting CNN features...")
    X_tensors, y_labels = build_training_tensors(train_csv)

    X_feats = []
    model.cnn.eval()
    with torch.no_grad():
        for x_t in X_tensors:
            B, T, C, H, W = x_t.shape
            x_reshaped = x_t.view(B * T, C, H, W).to(device)
            feats = model.cnn(x_reshaped).view(B, T, -1).cpu()
            X_feats.append(feats)

    X = torch.cat(X_feats, dim=0).to(device)
    y = torch.tensor(y_labels, dtype=torch.float32).unsqueeze(1).to(device)

    pos = int((y == 1.0).sum().item())
    neg = int((y == 0.0).sum().item())
    print(f"  Extracted {len(y)} clips (Accident: {pos}, Normal: {neg})")

    optimizer = optim.Adam(model.lstm.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(1, pos)]).to(device))

    print(f"\n[Stage 2/2] Training LSTM classifier for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.lstm.train()
        optimizer.zero_grad()
        logits = model.lstm(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 or epoch == epochs:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) >= 0.5).float()
                acc = (preds == y).float().mean().item()
            print(f"  Epoch {epoch:2d}/{epochs} | Loss: {loss.item():.4f} | Accuracy: {acc * 100:.1f}%")

    torch.save(model.state_dict(), checkpoint_path)
    print("=" * 65)
    print(f"[+] Saved best model checkpoint to: {checkpoint_path}")
    print("=" * 65)
    return model


if __name__ == "__main__":
    train_model(epochs=60)
