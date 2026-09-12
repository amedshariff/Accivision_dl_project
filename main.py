"""
Top-Level Main Entrypoint — Road Safety AI
Runs the complete 4-stage pipeline:
1. OpenCV Preprocessing
2. YOLO Object Detection
3. ByteTrack Object Tracking
4. CNN + LSTM Temporal Accident Classification -> Verdict (Accident / Normal)
"""
import sys
import argparse
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from integration.accident_detection_pipeline import AccidentDetectionPipeline

def main():
    parser = argparse.ArgumentParser(
        description="Road Safety AI — End-to-End Accident Detection Pipeline (OpenCV -> YOLO -> Tracking -> CNN-LSTM)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="data/raw/videos/accident/mock_accident.mp4",
        help="Path to input road video (mp4, avi, mov)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (useful for rapid testing)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom CNN+LSTM model weights checkpoint (.pth)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Root output directory"
    )

    args = parser.parse_args()

    pipeline = AccidentDetectionPipeline(
        model_checkpoint=args.model,
        output_base=args.output_dir
    )
    pipeline.run(
        video_path=args.video,
        max_frames=args.max_frames
    )

if __name__ == "__main__":
    main()
