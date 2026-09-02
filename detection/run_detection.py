"""
Run Detection CLI (Person 1 - Level 2)
Executes YOLO object detection on a road video and saves annotated video, CSV, and JSON deliverables.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detection.yolo_detector import YOLODetector
from detection.config import DEFAULT_OUTPUT_DIR, DEFAULT_CONFIDENCE, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Person 1 - Run YOLO Road Object Detection")
    parser.add_argument("--input", type=str, default="data/raw/real_road_test.mp4", help="Path to input video")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to YOLO model weights")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")

    args = parser.parse_args()

    print("=" * 65)
    print("PERSON 1 - YOLO ROAD OBJECT DETECTION")
    print("=" * 65)
    print(f"Input video: {args.input}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Output directory: {args.output_dir}")

    detector = YOLODetector(model_path=args.model, confidence_threshold=args.conf)
    summary = detector.process_video(
        video_path=args.input,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )

    print()
    print("=" * 65)
    print("PERSON 1 - DETECTION COMPLETED")
    print("=" * 65)
    print(f"Frames processed: {summary['frames_processed']}")
    print(f"Total detections: {summary['total_detections']}")
    print(f"Annotated video:  {summary['annotated_video']}")
    print(f"Detections CSV:   {summary['detections_csv']}")
    print(f"Detections JSON:  {summary['detections_json']}")


if __name__ == "__main__":
    main()
