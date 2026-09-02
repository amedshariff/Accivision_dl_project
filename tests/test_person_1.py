"""
Unit Tests for Person 1 (Video Preprocessing & YOLO Detection)
"""

import os
import unittest
import numpy as np
from pathlib import Path

from preprocessing.video_processor import VideoProcessor
from detection.yolo_detector import YOLODetector
from detection.config import MODEL_PATH, ROAD_CLASSES


class TestPerson1(unittest.TestCase):

    def setUp(self):
        self.sample_video = "data/raw/mock_normal.mp4"
        self.output_test_dir = Path("outputs/test_p1")
        self.output_test_dir.mkdir(parents=True, exist_ok=True)

    def test_video_info_valid(self):
        processor = VideoProcessor()
        info = processor.get_video_info(self.sample_video)
        self.assertIn("fps", info)
        self.assertIn("width", info)
        self.assertIn("height", info)
        self.assertIn("frame_count", info)
        self.assertGreater(info["fps"], 0)
        self.assertGreater(info["width"], 0)
        self.assertGreater(info["height"], 0)

    def test_video_info_invalid_path(self):
        processor = VideoProcessor()
        with self.assertRaises(FileNotFoundError):
            processor.get_video_info("non_existent_path.mp4")

    def test_video_processor_execution(self):
        processor = VideoProcessor(target_width=320, target_height=180)
        out_path = str(self.output_test_dir / "test_processed.mp4")
        res = processor.process_video(self.sample_video, out_path, max_frames=5)
        self.assertTrue(os.path.exists(out_path))
        self.assertEqual(res["frames_processed"], 5)
        self.assertEqual(res["width"], 320)
        self.assertEqual(res["height"], 180)

    def test_yolo_detector_schema(self):
        detector = YOLODetector(model_path=str(MODEL_PATH))
        dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        dets, annotated = detector.detect_frame(dummy_frame, frame_id=1, timestamp_sec=0.0)

        self.assertIsInstance(dets, list)
        self.assertEqual(annotated.shape, dummy_frame.shape)

        # If detections occur, verify schema fields
        for d in dets:
            self.assertIn("frame_id", d)
            self.assertIn("timestamp_sec", d)
            self.assertIn("class", d)
            self.assertIn("x1", d)
            self.assertIn("y1", d)
            self.assertIn("x2", d)
            self.assertIn("y2", d)
            self.assertIn("confidence", d)
            self.assertIn(d["class"], ROAD_CLASSES)


if __name__ == "__main__":
    unittest.main()
