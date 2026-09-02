"""
Unit Tests for Person 2 (Object Tracking & Motion Features)
"""

import unittest
from tracking.trajectory import TrajectoryTracker
from tracking.motion_features import MotionFeatureExtractor


class TestPerson2(unittest.TestCase):

    def test_trajectory_tracker_displacement(self):
        tracker = TrajectoryTracker(history_length=5)
        # First point
        disp1, total1 = tracker.update(track_id=1, center=(100, 100))
        self.assertEqual(disp1, 0.0)
        self.assertEqual(total1, 0.0)

        # Move 30px right, 40px down -> hypotenuse 50px
        disp2, total2 = tracker.update(track_id=1, center=(130, 140))
        self.assertEqual(disp2, 50.0)
        self.assertEqual(total2, 50.0)

        trail = tracker.get_trail(track_id=1)
        self.assertEqual(len(trail), 2)
        self.assertEqual(trail, [(100, 100), (130, 140)])

    def test_motion_feature_direction(self):
        extractor = MotionFeatureExtractor(fps=30.0, movement_threshold_px=5.0)

        # Stationary
        dir_stat = extractor.calculate_direction((100, 100), (102, 103))
        self.assertEqual(dir_stat, "Stationary")

        # Right
        dir_right = extractor.calculate_direction((100, 100), (130, 100))
        self.assertEqual(dir_right, "Right")

        # Left
        dir_left = extractor.calculate_direction((100, 100), (70, 100))
        self.assertEqual(dir_left, "Left")

        # Down
        dir_down = extractor.calculate_direction((100, 100), (100, 140))
        self.assertEqual(dir_down, "Down")

        # Up
        dir_up = extractor.calculate_direction((100, 100), (100, 60))
        self.assertEqual(dir_up, "Up")

    def test_motion_feature_speed(self):
        # 80 px/m, 30 fps, 80 px displacement in 1 frame -> 80 * 30 = 2400 px/s -> 30 m/s
        extractor = MotionFeatureExtractor(fps=30.0, pixels_per_meter=80.0)
        extractor.prev_positions[1] = (100, 100)
        features = extractor.compute_features(track_id=1, curr_pos=(180, 100), displacement_px=80.0)

        self.assertEqual(features["direction"], "Right")
        self.assertEqual(features["status"], "Moving")
        self.assertEqual(features["speed_mps"], 30.0)
        self.assertIn("HIGH_SPEED", features["alerts"])


if __name__ == "__main__":
    unittest.main()
