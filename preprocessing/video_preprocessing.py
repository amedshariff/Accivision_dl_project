"""
Video Preprocessing Pipeline Module
Connects frame extraction, resizing, and normalization for DL model input.
"""
from typing import List, Tuple
import numpy as np
import torch

from preprocessing.frame_extraction import extract_frames
from preprocessing.resize_frames import resize_frames
from preprocessing.normalize_frames import normalize_sequence
from preprocessing.utils import get_video_metadata

class VideoPreprocessor:
    def __init__(self, sequence_length: int = 16, frame_size: Tuple[int, int] = (224, 224)):
        self.sequence_length = sequence_length
        self.frame_size = frame_size

    def preprocess_video(self, video_path: str) -> Tuple[torch.Tensor, List[np.ndarray], dict]:
        """
        Extracts, resizes, and normalizes video frames for CNN-LSTM.
        Returns:
            tensor: (1, T, C, H, W) normalized tensor
            raw_resized_frames: list of resized BGR frames (for visualization)
            metadata: video metadata dict
        """
        meta = get_video_metadata(video_path)
        raw_frames = extract_frames(video_path, num_frames=self.sequence_length)
        resized_frames = resize_frames(raw_frames, target_size=self.frame_size)
        seq_tensor = normalize_sequence(resized_frames)  # (T, C, H, W)
        batch_tensor = seq_tensor.unsqueeze(0)  # (1, T, C, H, W)
        return batch_tensor, resized_frames, meta
