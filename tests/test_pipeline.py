"""
Integration Tests for Complete Person 1 + Person 2 Pipeline
Includes:
- Test 1: Valid Video Execution (real_road_test.mp4)
- Test 2: Different Video Execution (road_test.mp4)
- Test 3: Invalid / Missing Video Error Handling
"""

import os
import unittest
from pathlib import Path

from pipeline import AccidentDetectionPipeline


class TestPipelineIntegration(unittest.TestCase):

    def setUp(self):
        self.output_test_dir = Path("outputs/test_pipeline_run")
        self.pipeline = AccidentDetectionPipeline(output_base_dir=str(self.output_test_dir))

    def test_1_valid_video(self):
        """Test 1 — Valid video: runs Person 1 -> Person 2 -> Stop"""
        video_path = "data/raw/real_road_test.mp4"
        self.assertTrue(os.path.exists(video_path), f"Video missing: {video_path}")

        res = self.pipeline.run(video_path=video_path, max_frames=15)

        # Verify Person 1 outputs
        p1_proc = self.output_test_dir / "detections" / "processed_road_test.mp4"
        p1_annot = self.output_test_dir / "detections" / "annotated_road_test.mp4"
        p1_csv = self.output_test_dir / "detections" / "yolo_detections.csv"
        p1_json = self.output_test_dir / "detections" / "yolo_detections.json"

        self.assertTrue(p1_proc.exists(), "Processed video missing")
        self.assertTrue(p1_annot.exists(), "Annotated video missing")
        self.assertTrue(p1_csv.exists(), "Detections CSV missing")
        self.assertTrue(p1_json.exists(), "Detections JSON missing")

        # Verify Person 2 outputs
        p2_track = self.output_test_dir / "tracking" / "tracked_video.mp4"
        p2_csv = self.output_test_dir / "tracking" / "trajectories.csv"
        p2_json = self.output_test_dir / "tracking" / "motion_features.json"
        p2_rep = self.output_test_dir / "tracking" / "tracking_report.txt"

        self.assertTrue(p2_track.exists(), "Tracked video missing")
        self.assertTrue(p2_csv.exists(), "Trajectories CSV missing")
        self.assertTrue(p2_json.exists(), "Motion features JSON missing")
        self.assertTrue(p2_rep.exists(), "Tracking report missing")

    def test_2_different_video(self):
        """Test 2 — Different video: ensure pipeline is not hardcoded"""
        video_path = "data/raw/road_test.mp4"
        self.assertTrue(os.path.exists(video_path), f"Video missing: {video_path}")

        res = self.pipeline.run(video_path=video_path, max_frames=15)
        self.assertEqual(res["video_info"]["path"], video_path)
        self.assertGreater(res["person_1_preprocessing"]["frames_processed"], 0)

    def test_3_invalid_missing_input(self):
        """Test 3 — Invalid/missing video: handles error appropriately"""
        with self.assertRaises(FileNotFoundError):
            self.pipeline.run(video_path="data/raw/non_existent_file_xyz.mp4")


if __name__ == "__main__":
    unittest.main()
