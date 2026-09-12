"""
Accident Predictor Module (Inference)
Classifies road videos as ACCIDENT or NORMAL using the CNN + LSTM temporal model.
"""
import os
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from models.cnn_lstm_model import CNNLSTMModel
from preprocessing.video_preprocessing import VideoPreprocessor
from inference.inference_config import DEFAULT_MODEL_PATH, ACCIDENT_THRESHOLD, SEQUENCE_LENGTH
from inference.prediction_utils import smooth_probabilities

class AccidentPredictor:
    def __init__(
        self,
        model_path: Optional[str] = None,
        threshold: float = ACCIDENT_THRESHOLD,
        sequence_length: int = SEQUENCE_LENGTH,
        device: Optional[str] = None
    ):
        self.threshold = threshold
        self.sequence_length = sequence_length
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Initialize model architecture
        self.model = CNNLSTMModel(pretrained_cnn=True, freeze_cnn=True).to(self.device)
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        
        if self.model_path.exists():
            try:
                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                print(f"[AccidentPredictor] Loaded model checkpoint from: {self.model_path}")
            except Exception as e:
                print(f"[AccidentPredictor] Warning: Could not load checkpoint ({e}), using default weights.")
        else:
            print(f"[AccidentPredictor] Checkpoint not found at {self.model_path}. Using initialized model.")

        self.model.eval()
        self.preprocessor = VideoPreprocessor(sequence_length=self.sequence_length)

    def predict_video(self, video_path: str) -> Dict[str, Any]:
        """
        Accepts any road video and determines whether an accident occurs.
        Returns detailed prediction dictionary.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        batch_tensor, _, meta = self.preprocessor.preprocess_video(video_path)
        batch_tensor = batch_tensor.to(self.device)  # (1, T, C, H, W)

        with torch.no_grad():
            logits = self.model(batch_tensor)
            prob = torch.sigmoid(logits).item()

        is_accident = bool(prob >= self.threshold)
        verdict = "ACCIDENT DETECTED" if is_accident else "NORMAL TRAFFIC (NO ACCIDENT)"
        confidence_pct = round(prob * 100, 2)

        return {
            "video_path": str(video_path),
            "is_accident": is_accident,
            "verdict": verdict,
            "accident_probability": round(prob, 4),
            "confidence_percent": confidence_pct,
            "duration_sec": meta["duration_sec"],
            "total_frames": meta["frame_count"],
            "fps": meta["fps"]
        }
