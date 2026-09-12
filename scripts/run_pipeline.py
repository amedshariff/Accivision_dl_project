"""
Run Pipeline Script
"""
import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from integration.accident_detection_pipeline import AccidentDetectionPipeline

def main():
    parser = argparse.ArgumentParser(description="Run Accident Detection Pipeline")
    parser.add_argument("--video", type=str, required=True, help="Path to input road video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--model", type=str, default=None, help="Custom CNN+LSTM weights checkpoint")
    args = parser.parse_args()

    pipeline = AccidentDetectionPipeline(model_checkpoint=args.model)
    pipeline.run(video_path=args.video, max_frames=args.max_frames)

if __name__ == "__main__":
    main()
