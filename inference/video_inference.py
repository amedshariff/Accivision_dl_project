"""
Video Inference Visualizer Module (Inference)
Produces an annotated video displaying detection boxes, tracking IDs,
and a real-time HUD gauge showing the CNN-LSTM accident probability.
"""
import os
import cv2
from pathlib import Path
from typing import Dict, Any, Optional

from inference.accident_predictor import AccidentPredictor
from tracking.tracker import MultiObjectTracker

class VideoInferenceVisualizer:
    def __init__(
        self,
        predictor: Optional[AccidentPredictor] = None,
        tracker: Optional[MultiObjectTracker] = None,
        output_dir: str = "outputs/predictions"
    ):
        self.predictor = predictor or AccidentPredictor()
        self.tracker = tracker or MultiObjectTracker()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def process_and_visualize(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        # 1. Run holistic CNN+LSTM sequence prediction
        prediction = self.predictor.predict_video(video_path)
        prob = prediction["accident_probability"]
        is_accident = prediction["is_accident"]

        # 2. Render visual stream
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        video_name = Path(video_path).stem
        output_video_path = self.output_dir / f"predicted_{video_name}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (w, h))

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                frame_id += 1
                ts = (frame_id - 1) / fps

                # Run tracker for bounding boxes and motion trails
                _, annotated_frame = self.tracker.track_frame(
                    frame, frame_id=frame_id, timestamp_sec=ts, fps=fps, draw_annotations=True
                )

                # Draw Top HUD Banner
                hud_color = (0, 0, 220) if is_accident else (0, 180, 0)
                # Background bar
                cv2.rectangle(annotated_frame, (0, 0), (w, 55), (20, 20, 20), -1)
                cv2.line(annotated_frame, (0, 55), (w, 55), hud_color, 2)

                # Status text
                status_str = "STATUS: ACCIDENT DETECTED!" if is_accident else "STATUS: NORMAL TRAFFIC (SAFE)"
                cv2.putText(annotated_frame, status_str, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, hud_color, 2, cv2.LINE_AA)

                # Probability text & meter
                prob_str = f"Accident Prob: {prob * 100:.1f}%"
                cv2.putText(annotated_frame, prob_str, (w - 320, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                writer.write(annotated_frame)
                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()
            writer.release()

        prediction["annotated_output_video"] = str(output_video_path)
        return prediction
