"""
Track Objects Function Module
"""
from typing import Tuple, List, Dict, Any
import numpy as np
from tracking.tracker import MultiObjectTracker

def track_frame_objects(
    tracker: MultiObjectTracker,
    frame: np.ndarray,
    frame_id: int,
    timestamp_sec: float,
    fps: float = 30.0
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    return tracker.track_frame(frame, frame_id=frame_id, timestamp_sec=timestamp_sec, fps=fps)
