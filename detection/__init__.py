"""
Object Detection Package (Person 1 - Level 2)
"""
from detection.config import MODEL_PATH, ROAD_CLASSES, DEFAULT_CONFIDENCE
from detection.yolo_detector import YOLODetector

__all__ = ["YOLODetector", "MODEL_PATH", "ROAD_CLASSES", "DEFAULT_CONFIDENCE"]
