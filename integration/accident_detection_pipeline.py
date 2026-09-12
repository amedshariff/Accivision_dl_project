"""
Accident Detection Pipeline (Integration Module)
Executes all 4 components sequentially:
1. OpenCV Preprocessing
2. YOLO Road Object Detection
3. Multi-Object Tracking & Kinematics
4. CNN + LSTM Temporal Classification
"""
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.video_processor import VideoProcessor
from detection.yolo_detector import YOLODetector
from tracking.tracker import MultiObjectTracker
from inference.accident_predictor import AccidentPredictor
from inference.video_inference import VideoInferenceVisualizer

class AccidentDetectionPipeline:
    def __init__(self, model_checkpoint: Optional[str] = None, output_base: str = "outputs"):
        self.output_base = Path(output_base)
        self.video_processor = VideoProcessor()
        self.detector = YOLODetector()
        self.tracker = MultiObjectTracker()
        self.predictor = AccidentPredictor(model_path=model_checkpoint)
        self.visualizer = VideoInferenceVisualizer(
            predictor=self.predictor,
            tracker=self.tracker,
            output_dir=str(self.output_base / "predictions")
        )

    def run(self, video_path: str, max_frames: Optional[int] = None) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        print("\n" + "=" * 65)
        print("  ROAD SAFETY AI — FULL PIPELINE (OPENCV -> YOLO -> TRACKING -> CNN-LSTM)")
        print("=" * 65)

        # 1. OpenCV Preprocessing
        print("\n[Stage 1] OpenCV Preprocessing...")
        meta = self.video_processor.get_video_info(video_path)
        print(f"  Input: {video_path}")
        print(f"  Properties: {meta['width']}x{meta['height']} @ {meta['fps']} FPS | {meta['frame_count']} frames (~{meta['duration_sec']}s)")

        # 2. YOLO Detection & Tracking
        print("\n[Stage 2 & 3] YOLO Detection & Object Tracking...")
        tracking_out_dir = str(self.output_base / "tracked_videos")
        tracking_res = self.tracker.process_video(
            video_path=video_path,
            output_dir=tracking_out_dir,
            max_frames=max_frames
        )
        print(f"  Unique entities tracked: {tracking_res['unique_tracks']}")
        print(f"  Tracked deliverables:    {tracking_res['tracked_video']}")

        # 3. CNN + LSTM Temporal Classification
        print("\n[Stage 4] CNN + LSTM Temporal Accident Classifier...")
        pred_res = self.visualizer.process_and_visualize(video_path=video_path, max_frames=max_frames)

        # Final Verdict Banner
        print("\n" + "=" * 65)
        print("  FINAL ACCIDENT PREDICTION VERDICT")
        print("=" * 65)
        print(f"  VERDICT:              {pred_res['verdict']}")
        print(f"  ACCIDENT PROBABILITY: {pred_res['accident_probability'] * 100:.2f}%")
        print(f"  IS ACCIDENT:          {pred_res['is_accident']}")
        print(f"  OUTPUT VIDEO:         {pred_res['annotated_output_video']}")
        print("=" * 65)

        return {
            "metadata": meta,
            "tracking": tracking_res,
            "prediction": pred_res
        }
