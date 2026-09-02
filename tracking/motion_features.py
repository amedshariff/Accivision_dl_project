"""
Motion Features Module (Person 2 - Level 3)
Calculates direction, displacement, speed (px/s and m/s), movement status,
and identifies sudden kinematic changes (acceleration/direction changes).
"""

import math
from typing import Dict, Any, Tuple, Optional, List
from collections import defaultdict


class MotionFeatureExtractor:
    """
    Computes kinematics and motion attributes from tracked trajectories.
    """

    def __init__(
        self,
        fps: float = 30.0,
        pixels_per_meter: float = 80.0,
        movement_threshold_px: float = 4.0,
        abnormal_speed_mps: float = 20.0
    ):
        self.fps = fps if fps > 0 else 30.0
        self.pixels_per_meter = pixels_per_meter if pixels_per_meter > 0 else 80.0
        self.movement_threshold = movement_threshold_px
        self.abnormal_speed = abnormal_speed_mps

        # State tracking per track_id
        self.prev_positions: Dict[int, Tuple[int, int]] = {}
        self.prev_directions: Dict[int, str] = {}
        self.prev_speeds: Dict[int, float] = {}

        # Aggregate statistics per track_id
        self.track_stats: Dict[int, Dict[str, Any]] = defaultdict(lambda: {
            "observations": 0,
            "moving_count": 0,
            "stationary_count": 0,
            "speeds_mps": [],
            "max_speed_mps": 0.0,
            "directions": defaultdict(int),
            "abnormal_events": 0
        })

    def calculate_direction(
        self,
        prev_pos: Optional[Tuple[int, int]],
        curr_pos: Tuple[int, int]
    ) -> str:
        """
        Calculates approximate cardinal direction of movement.
        """
        if prev_pos is None:
            return "Stationary"

        dx = curr_pos[0] - prev_pos[0]
        dy = curr_pos[1] - prev_pos[1]

        if abs(dx) < self.movement_threshold and abs(dy) < self.movement_threshold:
            return "Stationary"

        if abs(dx) > abs(dy):
            return "Right" if dx > 0 else "Left"
        else:
            return "Down" if dy > 0 else "Up"

    def compute_features(
        self,
        track_id: int,
        curr_pos: Tuple[int, int],
        displacement_px: float
    ) -> Dict[str, Any]:
        """
        Calculates motion features for a single track update.
        """
        prev_pos = self.prev_positions.get(track_id)
        prev_dir = self.prev_directions.get(track_id)
        prev_speed = self.prev_speeds.get(track_id, 0.0)

        # Direction
        direction = self.calculate_direction(prev_pos, curr_pos)

        # Speed in pixels/sec and meters/sec using video frame rate
        speed_px_per_sec = displacement_px * self.fps
        speed_mps = speed_px_per_sec / self.pixels_per_meter

        # Movement status
        status = "Stationary" if direction == "Stationary" else "Moving"

        # Kinematic anomalies
        alerts: List[str] = []
        if speed_mps > self.abnormal_speed:
            alerts.append("HIGH_SPEED")

        if prev_dir and prev_dir != "Stationary" and direction != "Stationary" and prev_dir != direction:
            # Check opposite direction reversal (e.g. Left -> Right or Up -> Down)
            opposites = {("Left", "Right"), ("Right", "Left"), ("Up", "Down"), ("Down", "Up")}
            if (prev_dir, direction) in opposites:
                alerts.append("DIRECTION_REVERSAL")

        if prev_speed > 5.0 and (prev_speed - speed_mps) > 10.0:
            alerts.append("RAPID_DECELERATION")

        # Update stats
        stats = self.track_stats[track_id]
        stats["observations"] += 1
        if status == "Moving":
            stats["moving_count"] += 1
        else:
            stats["stationary_count"] += 1

        stats["speeds_mps"].append(speed_mps)
        if speed_mps > stats["max_speed_mps"]:
            stats["max_speed_mps"] = round(speed_mps, 2)
        stats["directions"][direction] += 1
        if alerts:
            stats["abnormal_events"] += 1

        # Cache previous state
        self.prev_positions[track_id] = curr_pos
        self.prev_directions[track_id] = direction
        self.prev_speeds[track_id] = speed_mps

        return {
            "displacement_px": round(displacement_px, 2),
            "direction": direction,
            "speed_px_s": round(speed_px_per_sec, 2),
            "speed_mps": round(speed_mps, 2),
            "status": status,
            "alerts": alerts
        }

    def get_summary_report(self) -> Dict[str, Any]:
        """
        Generates a summary report of all tracked objects.
        """
        summary = {}
        for track_id, stats in self.track_stats.items():
            obs = stats["observations"]
            speeds = stats["speeds_mps"]
            avg_speed = (sum(speeds) / len(speeds)) if speeds else 0.0
            dominant_dir = max(stats["directions"], key=stats["directions"].get) if stats["directions"] else "Unknown"

            summary[track_id] = {
                "observations": obs,
                "moving_count": stats["moving_count"],
                "stationary_count": stats["stationary_count"],
                "average_speed_mps": round(avg_speed, 2),
                "max_speed_mps": round(stats["max_speed_mps"], 2),
                "dominant_direction": dominant_dir,
                "abnormal_events": stats["abnormal_events"]
            }
        return summary
