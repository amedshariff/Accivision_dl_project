"""
Configuration for YOLO Object Detection (Person 1 - Level 2)
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "yolo11n.pt"

# Supported road-safety classes to detect and track
ROAD_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "person"
}

# Detection parameters
DEFAULT_CONFIDENCE = 0.40
DEFAULT_IOU_THRESHOLD = 0.45

# Output defaults
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "detections"
