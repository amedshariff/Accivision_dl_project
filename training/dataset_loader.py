"""
Dataset Loader Module (Training)
Loads video sequences for training CNN + LSTM accident classifiers.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Optional
from preprocessing.video_preprocessing import VideoPreprocessor

class VideoSequenceDataset(Dataset):
    def __init__(self, csv_split_path: str, sequence_length: int = 16, frame_size: Tuple[int, int] = (224, 224)):
        if not os.path.exists(csv_split_path):
            raise FileNotFoundError(f"Split file not found: {csv_split_path}")
        self.df = pd.read_csv(csv_split_path)
        self.preprocessor = VideoPreprocessor(sequence_length=sequence_length, frame_size=frame_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        video_path = row["video_path"]
        label = float(row["label"])
        
        # Extract and normalize sequence
        tensor, _, _ = self.preprocessor.preprocess_video(video_path)
        # tensor is (1, T, C, H, W) -> squeeze batch dim
        return tensor.squeeze(0), torch.tensor([label], dtype=torch.float32), video_path

def get_data_loader(csv_split_path: str, batch_size: int = 2, shuffle: bool = True) -> DataLoader:
    dataset = VideoSequenceDataset(csv_split_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
