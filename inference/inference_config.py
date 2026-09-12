"""
Inference Config Module
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "saved_models" / "best_model.pth"
ACCIDENT_THRESHOLD = 0.50
SEQUENCE_LENGTH = 16
