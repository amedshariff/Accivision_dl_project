"""
Accident Predictor Module (Inference)
Classifies road videos as ACCIDENT or NORMAL using the CNN + LSTM temporal model.
Supports multi-temporal window analysis to detect acute collisions anywhere in the video.
"""
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional

from models.cnn_lstm_model import CNNLSTMModel
from preprocessing.video_preprocessing import VideoPreprocessor
from preprocessing.normalize_frames import normalize_sequence
from preprocessing.resize_frames import resize_frames
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

    def _extract_window_tensor(self, video_path: str, start_f: int, end_f: int) -> torch.Tensor:
        """Extracts and normalizes a sequence of frames from a temporal interval [start_f, end_f]."""
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_f = max(0, start_f)
        end_f = min(total - 1, end_f)
        indices = np.linspace(start_f, end_f, self.sequence_length, dtype=int)
        indices_set = set(indices)

        frames = []
        curr = 0
        while True:
            ret, frame = cap.read()
            if not ret or curr > end_f:
                break
            if curr in indices_set:
                frames.append(frame)
            curr += 1
        cap.release()

        while len(frames) < self.sequence_length and len(frames) > 0:
            frames.append(frames[-1].copy())

        resized = resize_frames(frames, target_size=(224, 224))
        tensor = normalize_sequence(resized)  # (T, C, H, W)
        return tensor.unsqueeze(0).to(self.device)  # (1, T, C, H, W)

    def predict_video(
        self,
        video_path: str,
        kinematics_anomaly: bool = False,
        kinematics_score: float = 0.0
    ) -> Dict[str, Any]:
        """
        Accepts any road video and determines whether an accident occurs.
        Evaluates global temporal dynamics and sliding temporal sub-windows.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        # 1. Global video sequence evaluation
        batch_tensor, _, meta = self.preprocessor.preprocess_video(video_path)
        batch_tensor = batch_tensor.to(self.device)

        with torch.no_grad():
            logits_global = self.model(batch_tensor)
            prob_global = torch.sigmoid(logits_global).item()

        window_probs = [prob_global]
        timeline = [{"window": "global", "probability": round(prob_global, 4)}]
        total_frames = meta["frame_count"]
        fps = meta["fps"] or 30.0

        # 2. Multi-window analysis for longer videos to catch acute collision moments
        if total_frames > 35:
            win_size = min(total_frames - 1, max(30, int(fps * 2.5)))
            stride = max(15, int(fps * 1.0))
            for s in range(0, total_frames - win_size + 1, stride):
                e = s + win_size
                win_tensor = self._extract_window_tensor(video_path, s, e)
                with torch.no_grad():
                    p_win = torch.sigmoid(self.model(win_tensor)).item()
                window_probs.append(p_win)
                timeline.append({
                    "start_frame": s,
                    "end_frame": e,
                    "start_sec": round(s / fps, 2),
                    "end_sec": round(e / fps, 2),
                    "probability": round(p_win, 4)
                })

        # Peak model probability across temporal sequence
        max_model_prob = max(window_probs)

        # 3. Multi-modal fusion with kinematics / tracking (if provided)
        if kinematics_anomaly and kinematics_score > 0.3:
            final_prob = max(max_model_prob, 0.7 * max_model_prob + 0.3 * kinematics_score)
        else:
            final_prob = max_model_prob

        is_accident = bool(final_prob >= self.threshold)
        verdict = "ACCIDENT DETECTED" if is_accident else "NORMAL TRAFFIC (NO ACCIDENT)"
        confidence_pct = round(final_prob * 100, 2)

        return {
            "video_path": str(video_path),
            "is_accident": is_accident,
            "verdict": verdict,
            "accident_probability": round(final_prob, 4),
            "confidence_percent": confidence_pct,
            "model_probability": round(max_model_prob, 4),
            "duration_sec": meta["duration_sec"],
            "total_frames": meta["frame_count"],
            "fps": meta["fps"],
            "timeline": timeline
        }
