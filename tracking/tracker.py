"""
Multi-Object Tracker Module (Person 2 - Level 3)
Integrates ByteTrack with YOLO object detections to maintain persistent IDs across frames,
records trajectories, computes motion features, and renders visual trails and ID tags.
"""

import os
import csv
import json
import cv2
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from ultralytics import YOLO

from detection.config import MODEL_PATH, ROAD_CLASSES, DEFAULT_CONFIDENCE
from tracking.trajectory import TrajectoryTracker
from tracking.motion_features import MotionFeatureExtractor

DEFAULT_TRACKING_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "tracking"


class MultiObjectTracker:
    """
    Maintains persistent identities and kinematic state for road users across video frames.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        road_classes: Optional[set] = None,
        tracker_type: str = "bytetrack.yaml",
        history_length: int = 30,
        pixels_per_meter: float = 80.0
    ):
        path_to_use = str(model_path) if model_path else str(MODEL_PATH)
        if not os.path.exists(path_to_use):
            raise FileNotFoundError(f"Model file not found: {path_to_use}")

        self.model_path = path_to_use
        self.confidence_threshold = confidence_threshold
        self.road_classes = road_classes if road_classes is not None else ROAD_CLASSES
        self.tracker_type = tracker_type
        self.history_length = history_length
        self.pixels_per_meter = pixels_per_meter

        self.model = YOLO(self.model_path)
        self.trajectory_tracker = TrajectoryTracker(history_length=self.history_length)
        self.motion_extractor = MotionFeatureExtractor(pixels_per_meter=self.pixels_per_meter)

    def track_frame(
        self,
        frame: cv2.typing.MatLike,
        frame_id: int,
        timestamp_sec: float,
        fps: float = 30.0,
        draw_annotations: bool = True
    ) -> Tuple[List[Dict[str, Any]], cv2.typing.MatLike]:
        """
        Runs tracking on a single frame.
        Returns:
            track_records: List of track record dicts for the frame
            annotated_frame: Rendered frame with IDs, bounding boxes, labels, and trails
        """
        self.motion_extractor.fps = fps

        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            tracker=self.tracker_type,
            verbose=False
        )
        result = results[0]

        annotated_frame = frame.copy() if draw_annotations else frame
        records: List[Dict[str, Any]] = []

        if result.boxes is not None and result.boxes.id is not None and len(result.boxes) > 0:
            box_coords = result.boxes.xyxy.cpu().tolist()
            track_ids = result.boxes.id.int().cpu().tolist()
            class_ids = result.boxes.cls.int().cpu().tolist()
            confidences = result.boxes.conf.cpu().tolist()

            for box, track_id, cls_id, conf in zip(box_coords, track_ids, class_ids, confidences):
                cls_name = self.model.names[cls_id]
                if cls_name not in self.road_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                center = (center_x, center_y)

                # Update trajectory
                disp_px, total_dist_px = self.trajectory_tracker.update(track_id, center)

                # Compute motion features
                motion = self.motion_extractor.compute_features(track_id, center, disp_px)

                record = {
                    "frame_id": frame_id,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "track_id": track_id,
                    "class": cls_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "center_x": center_x,
                    "center_y": center_y,
                    "confidence": round(conf, 4),
                    "displacement_px": motion["displacement_px"],
                    "direction": motion["direction"],
                    "speed_mps": motion["speed_mps"],
                    "status": motion["status"],
                    "alerts": ";".join(motion["alerts"]) if motion["alerts"] else "Normal"
                }
                records.append(record)

                if draw_annotations:
                    # Draw bounding box
                    box_color = (0, 255, 0) if motion["status"] == "Moving" else (200, 200, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)

                    # Draw ID & Class banner
                    label_top = f"ID:{track_id} {cls_name}"
                    t_size = cv2.getTextSize(label_top, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, max(0, y1 - 20)),
                        (x1 + t_size[0] + 4, max(20, y1)),
                        box_color,
                        -1
                    )
                    cv2.putText(
                        annotated_frame,
                        label_top,
                        (x1 + 2, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

                    # Draw Motion attributes below box
                    label_bot = f"{motion['direction']} | {motion['speed_mps']:.1f} m/s"
                    cv2.putText(
                        annotated_frame,
                        label_bot,
                        (x1, min(frame.shape[0] - 5, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA
                    )

                    # Draw Center point
                    cv2.circle(annotated_frame, center, 4, (0, 0, 255), -1)

                    # Draw Trajectory trail
                    trail = self.trajectory_tracker.get_trail(track_id)
                    for i in range(1, len(trail)):
                        cv2.line(annotated_frame, trail[i - 1], trail[i], (255, 100, 0), 2)

        return records, annotated_frame

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_video: bool = True,
        save_csv: bool = True
    ) -> Dict[str, Any]:
        """
        Runs tracking across a full video, exporting deliverables to output_dir.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        out_dir = Path(output_dir) if output_dir else DEFAULT_TRACKING_OUTPUT_DIR
        out_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        tracked_video_path = out_dir / "tracked_video.mp4"
        trajectories_csv_path = out_dir / "trajectories.csv"
        motion_json_path = out_dir / "motion_features.json"
        report_path = out_dir / "tracking_report.txt"

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(tracked_video_path), fourcc, fps, (width, height))

        all_records: List[Dict[str, Any]] = []
        unique_ids = set()
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_id += 1
                timestamp_sec = round((frame_id - 1) / fps, 3)

                records, annotated_frame = self.track_frame(
                    frame,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    fps=fps,
                    draw_annotations=save_video
                )
                all_records.extend(records)
                for r in records:
                    unique_ids.add(r["track_id"])

                if writer:
                    # Watermark metadata
                    cv2.putText(
                        annotated_frame,
                        f"Frame: {frame_id} | Tracks: {len(unique_ids)}",
                        (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    writer.write(annotated_frame)

                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()

        # Save trajectories CSV
        if save_csv:
            with open(trajectories_csv_path, "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "frame_id", "timestamp_sec", "track_id", "class",
                    "x1", "y1", "x2", "y2", "center_x", "center_y",
                    "confidence", "displacement_px", "direction", "speed_mps", "status", "alerts"
                ])
                for r in all_records:
                    csv_writer.writerow([
                        r["frame_id"], r["timestamp_sec"], r["track_id"], r["class"],
                        r["x1"], r["y1"], r["x2"], r["y2"], r["center_x"], r["center_y"],
                        r["confidence"], r["displacement_px"], r["direction"], r["speed_mps"],
                        r["status"], r["alerts"]
                    ])

        # Generate summary report & JSON
        summary = self.motion_extractor.get_summary_report()
        with open(motion_json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        report_lines = [
            "=" * 60,
            "PERSON 2 - TRACKING & MOTION ANALYSIS REPORT",
            "=" * 60,
            f"Input video:        {video_path}",
            f"Frames processed:   {frame_id}/{total_frames}",
            f"Unique tracks:      {len(unique_ids)}",
            f"Total track points: {len(all_records)}",
            "",
            "PER-TRACK SUMMARY:",
            "-" * 60
        ]
        for tid, stat in summary.items():
            report_lines.append(
                f"Track ID {tid:3d} | Obs: {stat['observations']:4d} | "
                f"Avg Speed: {stat['average_speed_mps']:5.2f} m/s | "
                f"Max Speed: {stat['max_speed_mps']:5.2f} m/s | "
                f"Dir: {stat['dominant_direction']:10s} | "
                f"Moving: {stat['moving_count']:3d} | Stat: {stat['stationary_count']:3d}"
            )
        report_lines.append("=" * 60)

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines) + "\n")

        return {
            "video_path": video_path,
            "frames_processed": frame_id,
            "unique_tracks": len(unique_ids),
            "total_records": len(all_records),
            "tracked_video": str(tracked_video_path) if save_video else None,
            "trajectories_csv": str(trajectories_csv_path) if save_csv else None,
            "motion_json": str(motion_json_path),
            "report_path": str(report_path),
            "summary": summary
        }
