"""
YOLO Object Detection Module (Person 1 - Level 2)
Loads pretrained YOLO, detects road users (cars, motorcycles, buses, trucks, bicycles, people),
draws bounding boxes and labels, and generates structured CSV and JSON outputs.
"""

import os
import csv
import json
import cv2
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from ultralytics import YOLO

from detection.config import MODEL_PATH, ROAD_CLASSES, DEFAULT_CONFIDENCE, DEFAULT_OUTPUT_DIR


class YOLODetector:
    """
    Reusable detector leveraging pretrained YOLO for road object detection.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: float = DEFAULT_CONFIDENCE,
        road_classes: Optional[set] = None
    ):
        path_to_use = str(model_path) if model_path else str(MODEL_PATH)
        if not os.path.exists(path_to_use):
            raise FileNotFoundError(f"YOLO model weights not found at: {path_to_use}")

        self.model_path = path_to_use
        self.confidence_threshold = confidence_threshold
        self.road_classes = road_classes if road_classes is not None else ROAD_CLASSES
        self.model = YOLO(self.model_path)

    def detect_frame(
        self,
        frame: cv2.typing.MatLike,
        frame_id: int = 1,
        timestamp_sec: float = 0.0,
        draw_annotations: bool = True
    ) -> Tuple[List[Dict[str, Any]], cv2.typing.MatLike]:
        """
        Runs YOLO inference on a single frame.
        Returns:
            detections: List of detection dictionaries conforming to detection_schema.json
            annotated_frame: Frame with bounding boxes, labels, and confidence overlays
        """
        results = self.model.predict(
            frame,
            conf=self.confidence_threshold,
            verbose=False
        )
        result = results[0]
        detections: List[Dict[str, Any]] = []

        annotated_frame = frame.copy() if draw_annotations else frame

        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]

                if class_name not in self.road_classes:
                    continue

                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                record = {
                    "frame_id": frame_id,
                    "timestamp_sec": round(timestamp_sec, 3),
                    "class": class_name,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "confidence": round(confidence, 4)
                }
                detections.append(record)

                if draw_annotations:
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} {confidence:.2f}"
                    # Background tag for readability
                    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        annotated_frame,
                        (x1, max(0, y1 - 20)),
                        (x1 + t_size[0] + 4, max(20, y1)),
                        (0, 255, 0),
                        -1
                    )
                    cv2.putText(
                        annotated_frame,
                        label,
                        (x1 + 2, max(15, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA
                    )

        return detections, annotated_frame

    def process_video(
        self,
        video_path: str,
        output_dir: Optional[str] = None,
        max_frames: Optional[int] = None,
        save_video: bool = True,
        save_csv: bool = True,
        save_json: bool = True
    ) -> Dict[str, Any]:
        """
        Runs object detection on a full video and saves deliverables to output_dir.
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        out_dir = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
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

        annotated_video_path = out_dir / "annotated_road_test.mp4"
        csv_path = out_dir / "yolo_detections.csv"
        json_path = out_dir / "yolo_detections.json"

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

        all_detections: List[Dict[str, Any]] = []
        frame_id = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_id += 1
                timestamp_sec = round((frame_id - 1) / fps, 3)

                dets, annotated_frame = self.detect_frame(
                    frame,
                    frame_id=frame_id,
                    timestamp_sec=timestamp_sec,
                    draw_annotations=save_video
                )
                all_detections.extend(dets)

                if writer:
                    writer.write(annotated_frame)

                if max_frames and frame_id >= max_frames:
                    break
        finally:
            cap.release()
            if writer:
                writer.release()

        # Save CSV
        if save_csv:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow([
                    "frame_id", "timestamp_sec", "class",
                    "x1", "y1", "x2", "y2", "confidence"
                ])
                for d in all_detections:
                    csv_writer.writerow([
                        d["frame_id"], d["timestamp_sec"], d["class"],
                        d["x1"], d["y1"], d["x2"], d["y2"], d["confidence"]
                    ])

        # Save JSON
        if save_json:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(all_detections, f, indent=2)

        return {
            "video_path": video_path,
            "frames_processed": frame_id,
            "total_detections": len(all_detections),
            "annotated_video": str(annotated_video_path) if save_video else None,
            "detections_csv": str(csv_path) if save_csv else None,
            "detections_json": str(json_path) if save_json else None,
            "fps": fps,
            "width": width,
            "height": height
        }
