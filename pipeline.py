"""
Accident Detection System — Person 1 + Person 2 Integrated Pipeline
Pipeline:
  Video Input
       ↓
  Person 1: OpenCV Video Preprocessing (VideoProcessor)
       ↓
  Person 1: YOLO Object Detection (YOLODetector)
       ↓
  Person 2: ByteTrack Multi-Object Tracking & Motion Analysis (MultiObjectTracker)
       ↓
  Person 2 Final Output / Deliverables
       ↓
  [STOP HERE — Person 3 NOT IMPLEMENTED]
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Project root setup
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing.video_processor import VideoProcessor
from detection.yolo_detector import YOLODetector
from detection.config import MODEL_PATH, DEFAULT_CONFIDENCE
from tracking.tracker import MultiObjectTracker


class AccidentDetectionPipeline:
    """
    Executes Person 1 and Person 2 stages end-to-end.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        output_base_dir: Optional[str] = None
    ):
        self.base_dir = Path(output_base_dir) if output_base_dir else PROJECT_ROOT / "outputs"
        self.p1_output_dir = self.base_dir / "detections"
        self.p2_output_dir = self.base_dir / "tracking"

        self.p1_output_dir.mkdir(parents=True, exist_ok=True)
        self.p2_output_dir.mkdir(parents=True, exist_ok=True)

        self.model_path = model_path or str(MODEL_PATH)
        self.confidence_threshold = confidence_threshold

        self.video_processor = VideoProcessor()
        self.detector = YOLODetector(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold
        )
        self.tracker = MultiObjectTracker(
            model_path=self.model_path,
            confidence_threshold=self.confidence_threshold
        )

    def run(
        self,
        video_path: str,
        max_frames: Optional[int] = None,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Runs the complete Person 1 -> Person 2 pipeline on video_path.
        """
        print("=" * 70)
        print(" ACCIDENT DETECTION SYSTEM — PIPELINE (PERSON 1 + PERSON 2 ONLY)")
        print("=" * 70)

        # 0. Validation
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video file not found: {video_path}")

        info = self.video_processor.get_video_info(video_path)
        print(f"\n[Step 0] Input Video Validated:")
        print(f"  Path:       {video_path}")
        print(f"  Resolution: {info['width']}x{info['height']}")
        print(f"  FPS:        {info['fps']}")
        print(f"  Frames:     {info['frame_count']} (~{info['duration_sec']}s)")

        # 1. Person 1 - Level 1: OpenCV Preprocessing
        print(f"\n[Step 1] Person 1: OpenCV Video Preprocessing...")
        processed_video_path = str(self.p1_output_dir / "processed_road_test.mp4")
        if target_width and target_height:
            self.video_processor.target_width = target_width
            self.video_processor.target_height = target_height

        p1_prep_result = self.video_processor.process_video(
            input_path=video_path,
            output_path=processed_video_path,
            max_frames=max_frames
        )
        print(f"  Processed {p1_prep_result['frames_processed']} frames -> {processed_video_path}")

        # 2. Person 1 - Level 2: YOLO Object Detection
        print(f"\n[Step 2] Person 1: YOLO Object Detection...")
        p1_det_result = self.detector.process_video(
            video_path=processed_video_path,
            output_dir=str(self.p1_output_dir),
            max_frames=max_frames
        )
        print(f"  Detections:       {p1_det_result['total_detections']} objects found")
        print(f"  Annotated Video:  {p1_det_result['annotated_video']}")
        print(f"  Detections CSV:   {p1_det_result['detections_csv']}")
        print(f"  Detections JSON:  {p1_det_result['detections_json']}")

        # 3. Person 2 - Level 3: Object Tracking & Motion Analysis
        print(f"\n[Step 3] Person 2: ByteTrack Multi-Object Tracking & Motion Analysis...")
        p2_track_result = self.tracker.process_video(
            video_path=processed_video_path,
            output_dir=str(self.p2_output_dir),
            max_frames=max_frames
        )
        print(f"  Unique Tracks:    {p2_track_result['unique_tracks']}")
        print(f"  Track Records:    {p2_track_result['total_records']}")
        print(f"  Tracked Video:    {p2_track_result['tracked_video']}")
        print(f"  Trajectories CSV: {p2_track_result['trajectories_csv']}")
        print(f"  Motion JSON:      {p2_track_result['motion_json']}")
        print(f"  Tracking Report:  {p2_track_result['report_path']}")

        # 4. Final Handoff & Boundary Enforcement
        print("\n" + "=" * 70)
        print(" PERSON 2 COMPLETE — PIPELINE BOUNDARY REACHED")
        print("=" * 70)
        print("Status:")
        print("  [✓] Person 1 (OpenCV Preprocessing)       : COMPLETED")
        print("  [✓] Person 1 (YOLO Object Detection)      : COMPLETED")
        print("  [✓] Person 2 (ByteTrack Tracking)         : COMPLETED")
        print("  [✓] Person 2 (Motion & Trajectories CSV)  : COMPLETED")
        print("  [ ] Person 3 (CNN+LSTM Classifier)        : NOT IMPLEMENTED (Out of scope)")
        print("  [ ] Person 4 (Dashboard & Incident DB)    : NOT IMPLEMENTED (Out of scope)")
        print("=" * 70)

        return {
            "video_info": info,
            "person_1_preprocessing": p1_prep_result,
            "person_1_detection": p1_det_result,
            "person_2_tracking": p2_track_result
        }


def main():
    parser = argparse.ArgumentParser(description="Accident Detection System — Pipeline (Person 1 + Person 2)")
    parser.add_argument("--video", type=str, default="data/raw/real_road_test.mp4", help="Path to input road video")
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"), help="Base output directory")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Detection confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")

    args = parser.parse_args()

    pipeline = AccidentDetectionPipeline(
        confidence_threshold=args.conf,
        output_base_dir=args.output_dir
    )
    pipeline.run(
        video_path=args.video,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()
