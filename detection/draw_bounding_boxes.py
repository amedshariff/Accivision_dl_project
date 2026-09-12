"""
Draw Bounding Boxes Module
"""
import cv2
import numpy as np
from typing import List, Dict, Any

def draw_boxes(frame: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    annotated = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = d["x1"], d["y1"], d["x2"], d["y2"]
        label = f"{d['class']} {d['confidence']:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(annotated, label, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return annotated
