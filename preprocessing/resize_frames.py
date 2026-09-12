"""
Resize Frames Module (Preprocessing)
"""
import cv2
import numpy as np
from typing import List, Tuple

def resize_frame(frame: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """Resizes a single frame (width, height)."""
    return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)

def resize_frames(frames: List[np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> List[np.ndarray]:
    """Resizes a list of frames to the specified target dimensions."""
    return [resize_frame(f, target_size) for f in frames]
