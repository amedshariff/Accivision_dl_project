"""
Optical Flow Module (Preprocessing)
Calculates motion vectors between consecutive frames to detect sudden collision shock.
"""
import cv2
import numpy as np
from typing import List

def compute_optical_flow_magnitude(prev_frame: np.ndarray, next_frame: np.ndarray) -> float:
    """Computes average optical flow magnitude between two frames."""
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, next_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return float(np.mean(mag))

def compute_sequence_motion_energy(frames: List[np.ndarray]) -> List[float]:
    """Computes motion magnitude curve across a sequence of frames."""
    if len(frames) < 2:
        return [0.0]
    energies = [0.0]
    for i in range(1, len(frames)):
        energies.append(compute_optical_flow_magnitude(frames[i - 1], frames[i]))
    return energies
