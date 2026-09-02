"""
Run Tracking CLI (Person 2 - Level 3)
Executes ByteTrack multi-object tracking and motion feature extraction on a road video.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracking.tracker import MultiObjectTracker, DEFAULT_TRACKING_OUTPUT_DIR
from detection.config import DEFAULT_CONFIDENCE, MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description="Person 2 - Run Object Tracking & Motion Analysis")
    parser.add_argument("--input", type=str, default="data/raw/real_road_test.mp4", help="Path to input video")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_TRACKING_OUTPUT_DIR), help="Output directory")
    parser.add_argument("--model", type=str, default=str(MODEL_PATH), help="Path to YOLO weights")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE, help="Confidence threshold")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")

    args = parser.parse_args()

    print("=" * 65)
    print("PERSON 2 - OBJECT TRACKING & MOTION ANALYSIS")
    print("=" * 65)
    print(f"Input video:      {args.input}")
    print(f"Model:            {args.model}")
    print(f"Confidence:       {args.conf}")
    print(f"Output directory: {args.output_dir}")

    tracker = MultiObjectTracker(model_path=args.model, confidence_threshold=args.conf)
    summary = tracker.process_video(
        video_path=args.input,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )

    print()
    print("=" * 65)
    print("PERSON 2 - TRACKING COMPLETED")
    print("=" * 65)
    print(f"Frames processed:   {summary['frames_processed']}")
    print(f"Unique tracks:      {summary['unique_tracks']}")
    print(f"Total track points: {summary['total_records']}")
    print(f"Tracked video:      {summary['tracked_video']}")
    print(f"Trajectories CSV:   {summary['trajectories_csv']}")
    print(f"Motion JSON:        {summary['motion_json']}")
    print(f"Tracking Report:    {summary['report_path']}")


if __name__ == "__main__":
    main()
