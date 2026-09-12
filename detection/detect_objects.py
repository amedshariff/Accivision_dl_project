"""
Detect Objects Module
"""
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
from detection.yolo_detector import YOLODetector

def detect_objects_in_frame(detector: YOLODetector, frame: np.ndarray, frame_id: int = 1) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    return detector.detect_frame(frame, frame_id=frame_id)
