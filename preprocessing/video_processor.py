"""
Video Processor Module (Person 1 - Level 1)
Responsible for reading raw road videos, inspecting video properties,
validating integrity, consistent resizing/formatting, and saving processed videos.
"""

import os
import cv2
import argparse
from pathlib import Path
from typing import Dict, Any, Generator, Tuple, Optional


class VideoProcessor:
    """
    Reusable video processor for OpenCV-based frame ingestion and preprocessing.
    """

    def __init__(self, target_width: Optional[int] = None, target_height: Optional[int] = None):
        self.target_width = target_width
        self.target_height = target_height

    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """
        Inspects and returns metadata for the provided video.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (frame_count / fps) if fps > 0 else 0.0

        cap.release()

        return {
            "path": str(video_path),
            "fps": round(fps, 2) if fps > 0 else 30.0,
            "width": width,
            "height": height,
            "frame_count": frame_count,
            "duration_sec": round(duration_sec, 2),
        }

    def frame_generator(
        self,
        video_path: str,
        max_frames: Optional[int] = None
    ) -> Generator[Tuple[int, float, cv2.typing.MatLike], None, None]:
        """
        Generator yielding (frame_id, timestamp_sec, frame) sequentially.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_id = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_id += 1
                timestamp_sec = round((frame_id - 1) / fps, 3)

                if self.target_width and self.target_height:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))

                yield frame_id, timestamp_sec, frame

                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()

    def process_video(
        self,
        input_path: str,
        output_path: str,
        max_frames: Optional[int] = None,
        draw_overlays: bool = False
    ) -> Dict[str, Any]:
        """
        Processes video frame-by-frame and writes out the processed stream.
        """
        info = self.get_video_info(input_path)
        out_w = self.target_width or info["width"]
        out_h = self.target_height or info["height"]
        out_fps = info["fps"]

        output_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(output_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, out_fps, (out_w, out_h))
        if not writer.isOpened():
            raise IOError(f"Could not open VideoWriter for destination: {output_path}")

        processed_count = 0
        try:
            for frame_id, ts, frame in self.frame_generator(input_path, max_frames=max_frames):
                if draw_overlays:
                    cv2.putText(
                        frame,
                        f"Frame: {frame_id} | Time: {ts:.2f}s",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                writer.write(frame)
                processed_count += 1
        finally:
            writer.release()

        return {
            "input_path": input_path,
            "output_path": output_path,
            "frames_processed": processed_count,
            "width": out_w,
            "height": out_h,
            "fps": out_fps,
        }


def main():
    parser = argparse.ArgumentParser(description="Person 1 - OpenCV Video Preprocessor")
    parser.add_argument("--input", type=str, default="data/raw/real_road_test.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="outputs/detections/processed_road_test.mp4", help="Path to output processed video")
    parser.add_argument("--width", type=int, default=None, help="Target width (optional)")
    parser.add_argument("--height", type=int, default=None, help="Target height (optional)")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process (optional)")
    parser.add_argument("--overlay", action="store_true", help="Draw frame count overlay")

    args = parser.parse_args()

    processor = VideoProcessor(target_width=args.width, target_height=args.height)
    print(f"[Person 1] Inspecting video: {args.input}")
    info = processor.get_video_info(args.input)
    print(f"[Person 1] Properties: {info['width']}x{info['height']} @ {info['fps']} FPS, {info['frame_count']} frames ({info['duration_sec']}s)")

    print(f"[Person 1] Processing to: {args.output}")
    res = processor.process_video(args.input, args.output, max_frames=args.max_frames, draw_overlays=args.overlay)
    print(f"[Person 1] Completed! {res['frames_processed']} frames written to {res['output_path']}")


if __name__ == "__main__":
    main()
