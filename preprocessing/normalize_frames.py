"""
Normalize Frames Module (Preprocessing)
Applies standard ImageNet normalization for CNN backbone ingestion.
"""
import cv2
import torch
import numpy as np
from typing import List

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def normalize_frame_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """
    Converts a BGR OpenCV frame (H, W, C) to an ImageNet-normalized float tensor (C, H, W).
    """
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb_contiguous = np.ascontiguousarray(rgb)
    tensor = torch.from_numpy(rgb_contiguous).float() / 255.0  # (H, W, C) in [0, 1]
    tensor = tensor.permute(2, 0, 1)  # (C, H, W)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    normalized = (tensor - mean) / std
    return normalized

def normalize_sequence(frames: List[np.ndarray]) -> torch.Tensor:
    """
    Converts a list of T BGR frames to a normalized sequence tensor of shape (T, C, H, W).
    """
    tensors = [normalize_frame_tensor(f) for f in frames]
    return torch.stack(tensors, dim=0)
