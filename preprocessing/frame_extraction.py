"""
Frame Extraction Module (Preprocessing)
Extracts uniform temporal sequences of frames from videos.
"""
import cv2
import numpy as np
from typing import List, Optional

def extract_frames(video_path: str, num_frames: int = 16) -> List[np.ndarray]:
    """
    Extracts exactly num_frames uniformly distributed across the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        raise ValueError(f"Video {video_path} has 0 frames or is corrupt.")

    # Calculate uniform indices
    if total_frames < num_frames:
        indices = np.linspace(0, total_frames - 1, total_frames, dtype=int)
    else:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    indices_set = set(indices)
    frames = []
    current_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if current_idx in indices_set:
            frames.append(frame)
        current_idx += 1

    cap.release()

    # Handle edge cases (pad if fewer frames extracted)
    while len(frames) < num_frames and len(frames) > 0:
        frames.append(frames[-1].copy())

    if len(frames) == 0:
        raise ValueError(f"Failed to extract any valid frames from {video_path}")

    return frames[:num_frames]
